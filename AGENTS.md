# DSperate SF3000 notes

- SF3000/FrogUI input does **not** come from SDL joystick discovery. SDL reports
  zero joysticks on the target by design.
- Use pcsx4all's input contract: attach shared memory with
  `ftok("/tmp/joy_key", 'a')`, `shmget(key, 4, 0666)`, and `shmat()`. The
  shared `uint32_t` bit layout is defined in
  `pcsx4all/src/port/sf3000/port.c` (`CV_*` constants).
- Do not replace this with `SDL_JoystickOpen()` or GameController mappings.
  SDL keyboard/controller support is only for desktop fallback/testing.
