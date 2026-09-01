# Dataset audit

![boundary positions](boundary_positions.png)

Total records: **854**
Unique source articles: **798**

## Counts by domain x generator x construction type

| domain | generator | role | type1 single boundary | type2 single internal segment | type3 multiple internal segments | total |
|---|---|---|---|---|---|---|
| wikipedia | deepseek_v3 | seen | 131 | 115 | 106 | 352 |
| wikipedia | gemini_2_5_pro | held_out | 49 | 43 | 45 | 137 |
| wikipedia | gpt_4o | seen | 133 | 127 | 105 | 365 |

## Split balance

| generator | role | train | dev | test |
|---|---|---|---|---|
| deepseek_v3 | seen | 249 | 51 | 52 |
| gemini_2_5_pro | held_out | 0 | 62 | 75 |
| gpt_4o | seen | 253 | 59 | 53 |

Held-out generators appearing in train: **none (correct)**

## Sentence-level label balance

- Total sentences: **7935**
- AI sentences: **2844** (35.8%)
- Human sentences: **5091** (64.2%)

## Boundary-position distribution

Normalised position of the first boundary, by decile. A healthy dataset is spread across deciles; a spike in the first decile means position alone predicts the boundary.

| decile | range | count | share |
|---|---|---|---|
| 0 | 0.0-0.1 | 18 | 2.1% # |
| 1 | 0.1-0.2 | 87 | 10.2% ##### |
| 2 | 0.2-0.3 | 198 | 23.2% ########### |
| 3 | 0.3-0.4 | 163 | 19.1% ######### |
| 4 | 0.4-0.5 | 115 | 13.5% ###### |
| 5 | 0.5-0.6 | 140 | 16.4% ######## |
| 6 | 0.6-0.7 | 61 | 7.1% ### |
| 7 | 0.7-0.8 | 57 | 6.7% ### |
| 8 | 0.8-0.9 | 15 | 1.8%  |
| 9 | 0.9-1.0 | 0 | 0.0%  |

- min=0.083 p25=0.250 p50=0.364 p75=0.500 max=0.833
- First-decile share: 2.1% -> **OK - spread across the passage**

## Span-position distribution (Types 2 and 3)

| decile | count |
|---|---|
| 0 | 0 |
| 1 | 55 |
| 2 | 64 |
| 3 | 137 |
| 4 | 112 |
| 5 | 218 |
| 6 | 132 |
| 7 | 67 |
| 8 | 141 |
| 9 | 0 |

## Retry counts

| retries | records |
|---|---|
| 0 | 679 |
| 1 | 116 |
| 2 | 55 |
| 3 | 3 |
| 5 | 1 |

![by generator](boundary_positions_by_generator.png)