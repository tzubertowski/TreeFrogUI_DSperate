#include "display_hcge.h"
#include "core/gpu/gpu.h"
#include <dlfcn.h>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
namespace ds::sdl {
namespace { const char* path() {
  static char p[256]; if (p[0]) return p;
  int fd = ::open("/tmp/tfdevice.env", O_RDONLY); char b[512]{};
  if (fd >= 0) { const ssize_t n=::read(fd,b,sizeof(b)-1); ::close(fd); if (n>0) { const char* q=std::strstr(b,"TF_DRIVER="); if(q){q+=10; const char* e=std::strpbrk(q,"\r\n"); size_t z=e?size_t(e-q):std::strlen(q); if(z<sizeof(p)){std::memcpy(p,q,z);p[z]=0;}}}}
  if (!p[0]) std::strcpy(p,"/mnt/sdcard/cubegm/driver.so"); return p;
}}
bool HcgeOut::open() { using Init=int(*)(); handle_=::dlopen(path(),RTLD_NOW|RTLD_GLOBAL); if(!handle_) return false; auto init=(Init)::dlsym(handle_,"video_drivers_init"); deinit_=(Deinit)::dlsym(handle_,"video_driver_deinit"); present_=(Present)::dlsym(handle_,"video_driver_disp_frame"); if(!init||!deinit_||!present_||init()<=0){close();return false;} std::fprintf(stderr,"video: SF3000 HCGE\n"); return true; }
void HcgeOut::close(){if(handle_&&deinit_)deinit_();if(handle_)::dlclose(handle_);handle_=nullptr;deinit_=nullptr;present_=nullptr;}
void HcgeOut::present(const u32* const fb[2],const int screen[2],const int rect[2][4],const bool shown[2],u8 alpha,int w,int h){if(!present_||w<=0||h<=0)return;auto& d=frame_[page_++&1];d.assign(320*240,0);for(int i=0;i<2;++i)if(shown[i]){const int*r=rect[i];for(int y=0;y<r[3];++y)for(int x=0;x<r[2];++x){int dx=(r[0]+x)*320/w,dy=(r[1]+y)*240/h;if(dx<0||dx>=320||dy<0||dy>=240)continue;u32 p=fb[screen[i]][(y*192/r[3])*256+x*256/r[2]];u16 v=((p>>8)&0xf800)|((p>>5)&0x7e0)|((p>>3)&31);d[dy*320+dx]=v;}}present_(d.data(),320,240,640);}
}
