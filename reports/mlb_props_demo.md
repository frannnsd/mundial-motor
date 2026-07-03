# Demo props MLB (M3) — partido real de HOY

**Partido:** Cincinnati Reds @ Milwaukee Brewers — 2026-07-02, American Family Field
**as_of:** 2026-07-02 (point-in-time: solo cuentan juegos ANTERIORES a hoy)
**Lineups:** boxscore de HOY (game_pk 823765)
**Total de hits por equipo:** 8.5 puesto A MANO (el orquestador enchufa el total del cerebro unificado M1)

## Ks del abridor (prop principal)

| Abridor | Equipo | mu Ks | var | P(over 5.5) |
|---|---|---|---|---|
| Chase Burns | Cincinnati Reds | 6.99 | 8.14 | 67.8% |
| Jacob Misiorowski | Milwaukee Brewers | 8.18 | 9.34 | 80.7% |

## Cincinnati Reds — top 5 bateadores (vs Misiorowski)

Coherencia: suma de mu de los 9 = 8.5000000000 (== 8.5 exacto)

| # | Bateador | mu hits | P(1+ hit) | P(HR) |
|---|---|---|---|---|
| 1 | Elly De La Cruz | 1.13 | 67.8% | 15.5% |
| 2 | Sal Stewart | 1.05 | 65.2% | 17.6% |
| 7 | Edwin Arroyo | 0.97 | 62.2% | 7.2% |
| 6 | Noelvi Marte | 0.94 | 61.0% | 15.4% |
| 5 | Nathaniel Lowe | 0.94 | 60.8% | 13.7% |

## Milwaukee Brewers — top 5 bateadores (vs Burns)

Coherencia: suma de mu de los 9 = 8.5000000000 (== 8.5 exacto)

| # | Bateador | mu hits | P(1+ hit) | P(HR) |
|---|---|---|---|---|
| 2 | Jackson Chourio | 1.11 | 67.2% | 17.7% |
| 4 | William Contreras | 1.03 | 64.4% | 11.5% |
| 1 | Christian Yelich | 1.03 | 64.4% | 14.2% |
| 3 | Brice Turang | 1.00 | 63.1% | 13.2% |
| 7 | Sal Frelick | 0.92 | 60.2% | 7.1% |

---

Notas honestas:
- P(HR) es el prop MAS RUIDOSO: base rate ~3%/PA, el shrinkage (k=100 PAs) pesa mucho — señal débil, no pick.
- Dispersión de hits por bateador: Poisson(mu) (documentado); la de Ks: NegBin con Fano propio del pitcher (floor 1.0), via count_pmf del repo.
- Llamadas API usadas en esta corrida: 0 (cache-primero: re-correr = 0).
