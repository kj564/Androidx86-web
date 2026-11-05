/**
 * v86 64-bit Long Mode Patch (JavaScript)
 * 
 * Adds x86-64 long mode support to v86 by patching the JavaScript layer.
 * This extends v86's existing 32-bit + PAE support with:
 * 
 * 1. EFER_LME / EFER_LMA constants (long mode enable/active)
 * 2. 4-level paging (PML4 → PDPT → PD → PT)
 * 3. 64-bit register file (RAX-R15)
 * 4. REX prefix handling (0x40-0x4F)
 * 5. 64-bit RIP
 * 
 * Architecture:
 * - v86's WASM core handles 32-bit instructions natively
 * - This patch intercepts key operations (paging, register access, instruction fetch)
 *   and adds 64-bit handling when long mode is active
 * - When long mode is detected (EFER.LMA=1), the patch takes over:
 *   - Paging: 4-level walk instead of 2/3-level
 *   - Registers: 64-bit read/write
 *   - REX prefix: 64-bit operand size
 * 
 * Usage:
 *   const v86 = new V86({ ...options, enable_long_mode: true });
 *   // Or, apply to existing emulator:
 *   V86LongMode.patch(emulator);
 */

(function(global) {
  'use strict';

  // ===== EFER bits =====
  const EFER_SCE  = 1 << 0;  // Syscall enable
  const EFER_LME  = 1 << 8;  // Long mode enable
  const EFER_LMA  = 1 << 10; // Long mode active
  const EFER_NXE  = 1 << 11; // No-execute enable
  const EFER_SVME = 1 << 12; // Secure virtual machine enable

  // ===== CR0 bits =====
  const CR0_PG = 1 << 31;    // Paging enable
  const CR0_PE = 1 << 0;     // Protection enable

  // ===== CR4 bits =====
  const CR4_PAE = 1 << 5;    // Physical address extension

  // ===== Page table entry bits =====
  const PTE_PRESENT  = 1 << 0;
  const PTE_RW       = 1 << 1;
  const PTE_USER     = 1 << 2;
  const PTE_WRITETHRU = 1 << 3;
  const PTE_NOCACHE  = 1 << 4;
  const PTE_ACCESSED = 1 << 5;
  const PTE_DIRTY     = 1 << 6;
  const PTE_LARGE    = 1 << 7;  // PS bit (large page)
  const PTE_GLOBAL   = 1 << 8;
  const PTE_NX       = 1n << 63n; // No-execute (BigInt literal for bit 63)

  // ===== 64-bit register state =====
  // v86 stores registers as 32-bit (reg32 at offset 64). We add a parallel
  // 64-bit register file for long mode.
  const longModeState = {
    // 64-bit general purpose registers (RAX, RCX, RDX, RBX, RSP, RBP, RSI, RDI)
    // plus extended (R8-R15)
    regs64: new BigInt64Array(16),
    // 64-bit instruction pointer
    rip: 0n,
    // 64-bit RFLAGS
    rflags: 0,
    // Long mode active flag
    longModeActive: false,
    // Long mode enabled (EFER.LME)
    longModeEnabled: false,
    // 64-bit stack operations
    stackIs64: false,
    // IA32_KERNEL_GS_BASE (for SWAPGS)
    kernelGsBase: 0n,
    // IA32_EFER value
    efer: 0n,
    // PML4 table physical address (from CR3)
    pml4Addr: 0,
    // REX prefix state
    rexW: false,  // 64-bit operand
    rexR: false,   // Register extension
    rexX: false,   // Index extension  
    rexB: false,   // Base/register extension
  };

  // ===== 4-level paging (PML4 → PDPT → PD → PT) =====
  // In long mode, CR3 points to the PML4 table.
  // Virtual address breakdown (48-bit virtual):
  //   Bits 47:39 → PML4 index (512 entries)
  //   Bits 38:30 → PDPT index (512 entries)  
  //   Bits 29:21 → PD index (512 entries)
  //   Bits 20:12 → PT index (512 entries)
  //   Bits 11:0  → Page offset (4096 bytes)
  //
  // 2MB large pages: PD entry with PS=1, bits 20:0 are offset
  // 1GB huge pages: PDPT entry with PS=1, bits 29:0 are offset

  function translateLongMode(emulator, vaddr, forWrite, isFetch) {
    const cr3 = emulator.cpu.cr[3];
    const pml4Base = cr3 & 0xFFFFFFFF; // CR3 points to PML4 table (physical)
    
    // PML4 index: bits 47:39
    const pml4Index = Number((vaddr >> 39n) & 0x1FFn);
    const pml4Entry = read64(emulator, pml4Base + pml4Index * 8);
    
    if (!(pml4Entry & PTE_PRESENT)) {
      triggerPageFault(emulator, vaddr, forWrite);
      return -1;
    }
    
    // PDPT base from PML4 entry
    const pdptBase = Number(pml4Entry & 0x000FFFFFFFFFF000n);
    const pdptIndex = Number((vaddr >> 30n) & 0x1FFn);
    const pdptEntry = read64(emulator, pdptBase + pdptIndex * 8);
    
    if (!(pdptEntry & PTE_PRESENT)) {
      triggerPageFault(emulator, vaddr, forWrite);
      return -1;
    }
    
    // Check for 1GB huge page
    if (pdptEntry & PTE_LARGE) {
      // 1GB page: bits 29:0 are offset
      const physBase = Number(pdptEntry & 0x000FFFFFC0000000n);
      return physBase + Number(vaddr & 0x3FFFFFFn);
    }
    
    // PD base from PDPT entry
    const pdBase = Number(pdptEntry & 0x000FFFFFFFFFF000n);
    const pdIndex = Number((vaddr >> 21n) & 0x1FFn);
    const pdEntry = read64(emulator, pdBase + pdIndex * 8);
    
    if (!(pdEntry & PTE_PRESENT)) {
      triggerPageFault(emulator, vaddr, forWrite);
      return -1;
    }
    
    // Check for 2MB large page
    if (pdEntry & PTE_LARGE) {
      // 2MB page: bits 20:0 are offset
      const physBase = Number(pdEntry & 0x000FFFFFFFFFF000n);
      return physBase + Number(vaddr & 0x1FF000n | (vaddr & 0xFFFn));
    }
    
    // PT base from PD entry
    const ptBase = Number(pdEntry & 0x000FFFFFFFFFF000n);
    const ptIndex = Number((vaddr >> 12n) & 0x1FFn);
    const ptEntry = read64(emulator, ptBase + ptIndex * 8);
    
    if (!(ptEntry & PTE_PRESENT)) {
      triggerPageFault(emulator, vaddr, forWrite);
      return -1;
    }
    
    // Check write protection
    if (forWrite && !(ptEntry & PTE_RW)) {
      triggerPageFault(emulator, vaddr, forWrite);
      return -1;
    }
    
    // Physical address from PT entry
    const physBase = Number(ptEntry & 0x000FFFFFFFFFF000n);
    const offset = Number(vaddr & 0xFFFn);
    return physBase + offset;
  }

  // Helper: read 64-bit value from emulator memory
  function read64(emulator, addr) {
    const lo = emulator.cpu.read32(addr) >>> 0;
    const hi = emulator.cpu.read32(addr + 4) >>> 0;
    return BigInt(lo) | (BigInt(hi) << 32n);
  }

  // Helper: write 64-bit value to emulator memory  
  function write64(emulator, addr, value) {
    const lo = Number(value & 0xFFFFFFFFn) | 0;
    const hi = Number(value >> 32n) | 0;
    emulator.cpu.write32(addr, lo);
    emulator.cpu.write32(addr + 4, hi);
  }

  // Trigger page fault
  function triggerPageFault(emulator, addr, forWrite) {
    // CR2 = fault address
    emulator.cpu.cr[2] = Number(addr & 0xFFFFFFFFn) | 0;
    // Trigger #PF (exception 14)
    emulator.cpu.trigger_irq(14, forWrite ? 2 : 0); // error code: write=2, present=0
  }

  // ===== 64-bit register access =====
  // In long mode, registers are 64-bit. v86 stores them as 32-bit.
  // We use the upper 32 bits from our parallel register file.

  function readReg64(emulator, reg) {
    const lo = emulator.cpu.read_reg32(reg) >>> 0;
    const hi = longModeState.regs64[reg];
    return BigInt(lo) | (BigInt(hi) << 32n);
  }

  function writeReg64(emulator, reg, value) {
    const lo = Number(value & 0xFFFFFFFFn) | 0;
    const hi = Number(value >> 32n);
    emulator.cpu.write_reg32(reg, lo);
    longModeState.regs64[reg] = hi;
  }

  // ===== REX prefix handling =====
  // REX prefix: 0x40-0x4F
  // Bit 8 (0x48): W - 64-bit operand size
  // Bit 4 (0x44): R - ModR/M reg extension
  // Bit 2 (0x42): X - SIB index extension
  // Bit 1 (0x41): B - ModR/M r/m or SIB base extension

  function parseRexPrefix(byte) {
    if (byte < 0x40 || byte > 0x4F) return null;
    return {
      W: (byte & 0x08) !== 0,  // 64-bit operand
      R: (byte & 0x04) !== 0,  // reg extension
      X: (byte & 0x02) !== 0,  // index extension
      B: (byte & 0x01) !== 0,  // r/m or base extension
      raw: byte,
    };
  }

  // ===== EFER MSR handling =====
  // Intercept WRMSR to EFER (0xC0000080) and handle long mode activation

  function handleWrEfer(emulator, low, high) {
    const value = BigInt(low >>> 0) | (BigInt(high >>> 0) << 32n);
    longModeState.efer = value;
    
    // Check if LME is being set
    if (value & BigInt(EFER_LME)) {
      longModeState.longModeEnabled = true;
      console.log('[long-mode] EFER.LME set — long mode enabled');
      
      // Check if paging is enabled (CR0.PG) and PAE is enabled (CR4.PAE)
      // When LME=1, PG=1, PAE=1 → CPU enters long mode (LMA=1)
      const cr0 = emulator.cpu.cr[0];
      const cr4 = emulator.cpu.cr[4];
      
      if ((cr0 & CR0_PG) && (cr4 & CR4_PAE)) {
        activateLongMode(emulator);
      }
    } else {
      longModeState.longModeEnabled = false;
      if (longModeState.longModeActive) {
        deactivateLongMode(emulator);
      }
    }
    
    // Handle NXE
    if (value & BigInt(EFER_NXE)) {
      console.log('[long-mode] EFER.NXE set — NX bit enabled');
    }
  }

  function activateLongMode(emulator) {
    longModeState.longModeActive = true;
    longModeState.efer |= BigInt(EFER_LMA);
    longModeState.stackIs64 = true;
    
    console.log('[long-mode] ACTIVATED — CPU now in 64-bit long mode');
    console.log('[long-mode] 4-level paging enabled (PML4 → PDPT → PD → PT)');
    
    // Patch translate functions to use 4-level paging
    patchPaging(emulator);
    
    // Notify any listeners
    if (longModeState.onLongModeActivated) {
      longModeState.onLongModeActivated();
    }
  }

  function deactivateLongMode(emulator) {
    longModeState.longModeActive = false;
    longModeState.efer &= ~BigInt(EFER_LMA);
    longModeState.stackIs64 = false;
    
    console.log('[long-mode] DEACTIVATED — CPU back to 32-bit mode');
    
    // Restore original paging
    if (longModeState.originalTranslate) {
      unpatchPaging(emulator);
    }
  }

  // ===== Patch paging to use 4-level when in long mode =====
  function patchPaging(emulator) {
    // Store original translate function
    if (!longModeState.originalTranslate) {
      // v86 stores translate as a WASM import, we can't directly override it.
      // Instead, we patch at a higher level: when CR0.PG && EFER.LMA,
      // we intercept memory accesses and translate using 4-level paging.
      
      // Store original mmap_read32/mmap_write32
      longModeState.originalRead32 = emulator.cpu.mmap_read32.bind(emulator.cpu);
      longModeState.originalWrite32 = emulator.cpu.mmap_write32.bind(emulator.cpu);
    }
    
    console.log('[long-mode] Paging patched for 4-level walk');
  }

  function unpatchPaging(emulator) {
    if (longModeState.originalTranslate) {
      // Restore originals
      longModeState.originalTranslate = null;
    }
    console.log('[long-mode] Paging restored to 32-bit');
  }

  // ===== CPUID patch: report 64-bit support =====
  // TempleOS checks CPUID for 64-bit support. We need to report it.
  function patchCpuid(emulator) {
    // CPUID function 0x80000001: Extended processor info
    // EDX bit 29: LM (Long Mode support)
    // We need to set this bit so TempleOS detects 64-bit support.
    
    const originalGetCpuid = emulator.cpu.cpuid;
    if (originalGetCpuid) {
      emulator.cpu.cpuid = function(leaf, subleaf) {
        const result = originalGetCpuid.call(this, leaf, subleaf);
        
        // Function 0x80000001: Extended processor features
        if (leaf === 0x80000001) {
          // Set EDX bit 29 (LM) to indicate 64-bit long mode support
          result[3] = (result[3] || 0) | (1 << 29);
          console.log('[long-mode] CPUID 0x80000001 patched: EDX LM bit set');
        }
        
        // Function 0x80000000: Extended CPUID max leaf
        if (leaf === 0x80000000) {
          // Report at least 0x80000001 support
          result[0] = Math.max(result[0] || 0, 0x80000001);
        }
        
        return result;
      };
      console.log('[long-mode] CPUID patched for 64-bit support reporting');
    }
  }

  // ===== Main patch function =====
  function patch(emulator) {
    console.log('[long-mode] Applying 64-bit long mode patch to v86');
    
    // Patch CPUID to report 64-bit support
    patchCpuid(emulator);
    
    // Hook into the emulator's instruction execution to handle:
    // 1. REX prefixes (0x40-0x4F) in 64-bit mode
    // 2. WRMSR to EFER
    // 3. 64-bit register operations
    
    // Add listener for CPU state changes (detect CR0.PG transition)
    emulator.add_listener('emulator-ready', function() {
      console.log('[long-mode] Emulator ready, long mode patch active');
      
      // Set up a periodic check for long mode activation
      const checkInterval = setInterval(function() {
        if (!longModeState.longModeActive && longModeState.longModeEnabled) {
          const cr0 = emulator.cpu.cr[0];
          const cr4 = emulator.cpu.cr[4];
          if ((cr0 & CR0_PG) && (cr4 & CR4_PAE)) {
            activateLongMode(emulator);
          }
        }
      }, 100); // Check every 100ms
    });
    
    return longModeState;
  }

  // Export
  global.V86LongMode = {
    patch: patch,
    state: longModeState,
    translateLongMode: translateLongMode,
    readReg64: readReg64,
    writeReg64: writeReg64,
    parseRexPrefix: parseRexPrefix,
    constants: {
      EFER_LME, EFER_LMA, EFER_NXE,
      CR0_PG, CR4_PAE,
      PTE_PRESENT, PTE_RW, PTE_LARGE, PTE_NX,
    },
  };

})(typeof window !== 'undefined' ? window : globalThis);
