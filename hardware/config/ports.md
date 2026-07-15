# Port mapping — SO-101 arms

Determined 2026-07-15 (Session 3) by USB-C unplug test. **Use the stable `by-id`
serial**, not `/dev/ttyACMx` — the ACM numbers can reshuffle on reboot/replug.

| Role | Stable serial | by-id symlink | Enumerated (15 Jul) | PSU |
|---|---|---|---|---|
| **Leader** | `5B61036522` | `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B61036522-if00` | `/dev/ttyACM0` | **5V / 6A** |
| **Follower** | `5B3D046621` | `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B3D046621-if00` | `/dev/ttyACM1` | **12V / 8A** |

- Both are QinHeng CH340-type USB-serial (`1a86:55d3`). Identical chips → identity is
  the serial only.
- User in `dialout` → no sudo needed for serial access.
- **Camera** (ARC International, `05a3:9230`) → `/dev/video4` + `/dev/video5`. Built-in
  Acer cam = `/dev/video0–3`. See `cameras.md` once mounted (still capped 15 Jul).
