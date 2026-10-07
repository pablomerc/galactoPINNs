# Diagnostic runs: χ-stop by region, and a noisy Sun anchor

**Gate**: exact-anchor diagnostic runs reproduce the v2 runs' χ-stop epoch and rec4 for 30/30 models.

## χ-stop vs the best epoch of each region (exact anchor, observer-referenced error)

Excess = error at the χ-stop / best error along the trajectory − 1 (median [IQR] over 6 trials); epochs are medians.

| n | χ-stop epoch | star particles <4 kpc (rec4): best epoch, excess | star particles <15 kpc: best epoch, excess | bubble R=0.5 kpc: best epoch, excess | bubble R=2 kpc: best epoch, excess | bubble R=4 kpc: best epoch, excess |
|---|---|---|---|---|---|---|
| 50 | 25 | 14, +13% [+8, +28] | 0, +21% [+5, +59] | 216, +14% [+8, +17] | 52, +16% [+11, +25] | 18, +25% [+19, +35] |
| 200 | 142 | 131, +36% [+28, +40] | 0, +57% [+23, +67] | 207, +21% [+9, +29] | 108, +17% [+13, +19] | 186, +44% [+26, +55] |
| 500 | 538 | 112, +48% [+34, +60] | 0, +30% [+23, +66] | 961, +23% [+18, +36] | 162, +36% [+32, +41] | 96, +57% [+42, +70] |
| 1000 | 848 | 106, +92% [+88, +122] | 0, +54% [+38, +61] | 1081, +30% [+29, +36] | 108, +50% [+32, +63] | 102, +105% [+84, +133] |
| 2000 | 799 | 138, +94% [+69, +102] | 0, +46% [+35, +50] | 1304, +20% [+18, +28] | 372, +42% [+33, +50] | 96, +78% [+70, +94] |

## Exact vs noisy Sun anchor (rms error 10% of |a_sun|), identical draws and inits

Values at the χ-stop, median [IQR]. `g-` errors are observer-referenced (blind to the constant offset); absolute errors include it. Sun offset = |a_model(x_sun) − a_sun,true|.

| n | Sun measurement error [mm/s/yr] | Sun offset exact / noisy [mm/s/yr] | g-acc R=2 exact / noisy [%] | acc R=2 exact / noisy [%] | rec4 exact / noisy [%] | g-rec15 exact / noisy [%] | χ-stop epoch exact / noisy |
|---|---|---|---|---|---|---|---|
| 50 | 0.64 [0.48, 0.78] | 0.44 [0.15, 0.67] / 0.69 [0.54, 0.76] | 9.73 [8.55, 9.91] / 9.28 [8.43, 9.88] | 10.73 [8.59, 11.32] / 11.43 [10.54, 12.58] | 12.71 [12.61, 16.85] / 13.50 [12.66, 14.96] | 19.63 [16.66, 25.62] / 20.16 [16.96, 22.02] | 25 / 31 |
| 200 | 0.64 [0.48, 0.78] | 0.21 [0.15, 0.27] / 0.43 [0.30, 0.68] | 6.51 [5.60, 7.41] / 6.18 [5.77, 7.62] | 6.94 [6.03, 7.93] / 8.14 [7.25, 10.76] | 11.39 [9.60, 12.61] / 11.01 [9.54, 13.92] | 25.42 [19.52, 27.09] / 28.66 [21.01, 28.99] | 142 / 136 |
| 500 | 0.64 [0.48, 0.78] | 0.06 [0.04, 0.07] / 0.76 [0.40, 0.90] | 6.57 [5.87, 7.49] / 6.69 [5.85, 7.44] | 6.65 [5.89, 8.03] / 12.26 [9.53, 13.96] | 11.61 [8.68, 14.13] / 13.21 [12.98, 16.93] | 21.09 [19.84, 26.91] / 22.40 [20.79, 25.67] | 538 / 525 |
| 1000 | 0.64 [0.48, 0.78] | 0.07 [0.05, 0.11] / 0.66 [0.49, 0.75] | 6.44 [5.39, 6.62] / 6.43 [5.45, 6.60] | 6.60 [6.35, 7.25] / 11.85 [8.38, 12.75] | 11.45 [11.16, 12.47] / 14.54 [12.14, 15.38] | 24.96 [21.81, 26.08] / 22.53 [21.92, 23.51] | 848 / 845 |
| 2000 | 0.64 [0.48, 0.78] | 0.11 [0.05, 0.27] / 0.63 [0.50, 0.83] | 5.38 [4.62, 5.72] / 5.17 [4.49, 5.88] | 6.17 [4.67, 7.05] / 10.06 [9.22, 13.17] | 10.81 [9.96, 11.04] / 12.79 [11.63, 15.50] | 23.67 [21.56, 24.31] / 23.69 [20.76, 25.66] | 799 / 822 |

Per model: Sun offset of the noisy-anchor model vs its Sun measurement error (ratio 1 = the model's absolute field is off by exactly the measurement error):

- n = 50: ratio median 0.99 (range 0.77–3.79)
- n = 200: ratio median 0.84 (range 0.46–1.06)
- n = 500: ratio median 1.01 (range 0.72–4.05)
- n = 1000: ratio median 1.01 (range 0.62–1.05)
- n = 2000: ratio median 1.05 (range 0.93–1.48)
