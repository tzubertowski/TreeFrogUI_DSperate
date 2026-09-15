.set push
.set noreorder
.set noat
.text
.globl ds_jit_mips_make_code_visible
.ent ds_jit_mips_make_code_visible
ds_jit_mips_make_code_visible:
  rdhwr $1, $1
  beq $1, $0, 2f
  addiu $2, $1, -1
  beq $5, $0, 2f
  addu $5, $4, $5
  nor $2, $2, $0
  and $4, $4, $2
1:
  synci 0($4)
  addu $4, $4, $1
  sltu $2, $4, $5
  bne $2, $0, 1b
  nop
  sync
2:
  jr.hb $31
  nop
.end ds_jit_mips_make_code_visible
.set pop
