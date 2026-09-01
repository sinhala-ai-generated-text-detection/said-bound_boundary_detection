# Dataset audit

![boundary positions](boundary_positions.png)

Total records: **116**
Unique source articles: **84**

## Counts by domain x generator x construction type

| domain | generator | role | type1 single boundary | type2 single internal segment | type3 multiple internal segments | total |
|---|---|---|---|---|---|---|
| wikipedia | deepseek_v3 | seen | 12 | 15 | 11 | 38 |
| wikipedia | gemini_2_5_pro | held_out | 16 | 12 | 11 | 39 |
| wikipedia | gpt_4o | seen | 14 | 13 | 12 | 39 |

## Split balance

| generator | role | train | dev | test |
|---|---|---|---|---|
| deepseek_v3 | seen | 28 | 5 | 5 |
| gemini_2_5_pro | held_out | 0 | 20 | 19 |
| gpt_4o | seen | 27 | 7 | 5 |

Held-out generators appearing in train: **none (correct)**

## Sentence-level label balance

- Total sentences: **1082**
- AI sentences: **386** (35.7%)
- Human sentences: **696** (64.3%)

## Boundary-position distribution

Normalised position of the first boundary, by decile. A healthy dataset is spread across deciles; a spike in the first decile means position alone predicts the boundary.

| decile | range | count | share |
|---|---|---|---|
| 0 | 0.0-0.1 | 4 | 3.4% # |
| 1 | 0.1-0.2 | 13 | 11.2% ##### |
| 2 | 0.2-0.3 | 29 | 25.0% ############ |
| 3 | 0.3-0.4 | 22 | 19.0% ######### |
| 4 | 0.4-0.5 | 9 | 7.8% ### |
| 5 | 0.5-0.6 | 20 | 17.2% ######## |
| 6 | 0.6-0.7 | 9 | 7.8% ### |
| 7 | 0.7-0.8 | 8 | 6.9% ### |
| 8 | 0.8-0.9 | 2 | 1.7%  |
| 9 | 0.9-1.0 | 0 | 0.0%  |

- min=0.083 p25=0.222 p50=0.333 p75=0.500 max=0.833
- First-decile share: 3.4% -> **OK - spread across the passage**

## Span-position distribution (Types 2 and 3)

| decile | count |
|---|---|
| 0 | 0 |
| 1 | 9 |
| 2 | 11 |
| 3 | 17 |
| 4 | 18 |
| 5 | 28 |
| 6 | 16 |
| 7 | 9 |
| 8 | 19 |
| 9 | 0 |

## Retry counts

| retries | records |
|---|---|
| 0 | 95 |
| 1 | 14 |
| 2 | 7 |

![by generator](boundary_positions_by_generator.png)