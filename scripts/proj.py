D1 = 3.0                      # box 2's picks/token at B=1
dedup = {1:1.00, 2:1.66, 4:2.73, 5:3.20, 6:3.62, 8:4.42}   # MEASURED, routing trace
SERIAL, B1_MOE, B2_LEG = 35.9, 18.2, 35.0                  # MEASURED 71 ms floor decomposition
# calibrate box-2 per-layer overhead (link + wake-up + hub gap) so leg(1) == 35 ms
ovh = B2_LEG*1000/40 - 20*1 - 87*D1
print(f"box-2 per-layer overhead (B-independent): {ovh:.0f} us\n")
print(f"{'B':>2} {'D/layer':>8} {'box2 leg':>9} {'box1 moe':>9} {'step ms':>8} "
      f"{'@E=4.13':>8} {'@E=3.0':>7} {'serial+25%':>11}")
for B, r in dedup.items():
    D = D1*r
    leg = 40*(ovh + 20*B + 87*D)/1000
    moe = B1_MOE*r
    step = SERIAL + max(moe, leg)
    step45 = SERIAL*1.25 + max(moe, leg)
    print(f"{B:2d} {D:8.2f} {leg:9.1f} {moe:9.1f} {step:8.1f} "
          f"{4.13/step*1000:8.1f} {3.0/step*1000:7.1f} {4.13/step45*1000:11.1f}")
