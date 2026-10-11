# DSperate SF3000 notes

## R36HD deployment

- Build the target package from this repo with `./build_sf3000.sh`.
- Deploy with the guarded TreeFrogUI card deployer, not `copy_to_sd.sh` or a
  manual copy into `/run/media`: `cd /home/tomaszz/sf3000-work/sf3000_treefrogui`
  then `pkexec ./deploy.sh r36hd dsperate`.
- The `dsperate` payload copies `dsperate/build/sf3000-package/` to
  `cubegm/dsperate/`, verifies the package, and safely unmounts the card. Wait
  for the deployer’s “safe to remove” confirmation before ejecting it.
- If that payload is rejected, update `sf3000_treefrogui/deploy_device.sh` to
  support `dsperate`; do not bypass the card-label, VFAT, filesystem-check, and
  SHA-256 safeguards with ad-hoc copies.

- SF3000/FrogUI input does **not** come from SDL joystick discovery. SDL reports
  zero joysticks on the target by design.
- Use pcsx4all's input contract: attach shared memory with
  `ftok("/tmp/joy_key", 'a')`, `shmget(key, 4, 0666)`, and `shmat()`. The
  shared `uint32_t` bit layout is defined in
  `pcsx4all/src/port/sf3000/port.c` (`CV_*` constants).
- Do not replace this with `SDL_JoystickOpen()` or GameController mappings.
  SDL keyboard/controller support is only for desktop fallback/testing.
