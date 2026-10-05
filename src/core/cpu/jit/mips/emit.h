#pragma once
#include "core/types.h"
#include <cassert>
#include <cstring>
namespace ds::jit {
class MipsEmitter {
  u8* p_; size_t cap_, n_;
public:
  MipsEmitter(u8* p,size_t c):p_(p),cap_(c),n_(0){}
  size_t size() const{return n_;} u8* cur() const{return p_+n_;}
  void w(u32 x){assert(n_+4<=cap_); std::memcpy(p_+n_,&x,4); n_+=4;}
  static void patch(u8* p,u32 x){std::memcpy(p,&x,4);}
  void lui(u32 r,u32 x){w(0x3c000000u|(r<<16)|(x&0xffff));}
  void ori(u32 d,u32 s,u32 x){w(0x34000000u|(s<<21)|(d<<16)|(x&0xffff));}
  void addiu(u32 d,u32 s,s32 x){w(0x24000000u|(s<<21)|(d<<16)|(static_cast<u32>(x)&0xffff));}
  void addu(u32 d,u32 s,u32 t){w(0x00000021u|(s<<21)|(t<<16)|(d<<11));}
  void sll(u32 d,u32 s,u32 sh){w((s<<16)|(d<<11)|(sh<<6));}
  void srl(u32 d,u32 s,u32 sh){w(0x00000002u|(s<<16)|(d<<11)|(sh<<6));}
  void sra(u32 d,u32 s,u32 sh){w(0x00000003u|(s<<16)|(d<<11)|(sh<<6));}
  void andi(u32 d,u32 s,u32 x){w(0x30000000u|(s<<21)|(d<<16)|(x&0xffff));}
  void lbu(u32 d,s32 o,u32 s){w(0x90000000u|(s<<21)|(d<<16)|(static_cast<u32>(o)&0xffff));}
  void subu(u32 d,u32 s,u32 t){w(0x00000023u|(s<<21)|(t<<16)|(d<<11));}
  void and_(u32 d,u32 s,u32 t){w(0x00000024u|(s<<21)|(t<<16)|(d<<11));}
  void or_(u32 d,u32 s,u32 t){w(0x00000025u|(s<<21)|(t<<16)|(d<<11));}
  void xor_(u32 d,u32 s,u32 t){w(0x00000026u|(s<<21)|(t<<16)|(d<<11));}
  void nor(u32 d,u32 s,u32 t){w(0x00000027u|(s<<21)|(t<<16)|(d<<11));}
  void move(u32 d,u32 s){addu(d,s,0);}
  void lw(u32 d,s32 o,u32 s){w(0x8c000000u|(s<<21)|(d<<16)|(static_cast<u32>(o)&0xffff));}
  void sw(u32 d,s32 o,u32 s){w(0xac000000u|(s<<21)|(d<<16)|(static_cast<u32>(o)&0xffff));}
  void sb(u32 d,s32 o,u32 s){w(0xa0000000u|(s<<21)|(d<<16)|(static_cast<u32>(o)&0xffff));}
  void jr(u32 r){w(0x00000008u|(r<<21));} void jalr(u32 d,u32 r){w(0|(r<<21)|(d<<11)|9);}
  void nop(){w(0);}
  void j(u32 target){w(0x08000000u|((target>>2)&0x03ffffff));}
  size_t bnez(u32 r){size_t p=n_; w(0x14000000u|(r<<21)); return p;}
  size_t beq(u32 s,u32 t){size_t p=n_; w(0x10000000u|(s<<21)|(t<<16)); return p;}
  size_t b(){size_t p=n_; w(0x10000000u); return p;}
  size_t bltz(u32 r){size_t p=n_; w(0x04000000u|(r<<21)); return p;}
  size_t blez(u32 r){size_t p=n_; w(0x18000000u|(r<<21)); return p;}
  void patch_branch(size_t p,size_t target){s32 d=static_cast<s32>((target-(p+4))/4);u32 x;std::memcpy(&x,p_+p,4);x=(x&0xffff0000u)|(static_cast<u32>(d)&0xffffu);std::memcpy(p_+p,&x,4);}
  void load_ptr(u32 r,const void* p){uintptr_t x=reinterpret_cast<uintptr_t>(p);lui(r,x>>16);ori(r,r,x);}
  void patch_ptr(size_t p,u32 r,const void* q){uintptr_t x=reinterpret_cast<uintptr_t>(q);patch(p_+p,0x3c000000u|(r<<16)|(x>>16));patch(p_+p+4,0x34000000u|(r<<21)|(r<<16)|(x&0xffff));}
};
}
