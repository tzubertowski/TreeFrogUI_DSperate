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
  void lw(u32 d,s32 o,u32 s){w(0x8c000000u|(s<<21)|(d<<16)|(static_cast<u32>(o)&0xffff));}
  void sw(u32 d,s32 o,u32 s){w(0xac000000u|(s<<21)|(d<<16)|(static_cast<u32>(o)&0xffff));}
  void jr(u32 r){w(0x00000008u|(r<<21));} void jalr(u32 d,u32 r){w(0|(r<<21)|(d<<11)|9);}
  void nop(){w(0);}
  void j(u32 target){w(0x08000000u|((target>>2)&0x03ffffff));}
};
}
