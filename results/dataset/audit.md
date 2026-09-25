# Dataset audit

![boundary positions](boundary_positions.png)

Total records: **4244**
Unique source articles: **2630**

## Counts by domain x generator x construction type

| domain | generator | role | type1 single boundary | type2 single internal segment | type3 multiple internal segments | total |
|---|---|---|---|---|---|---|
| wikipedia | deepseek_v3 | seen | 650 | 576 | 535 | 1761 |
| wikipedia | gemini_2_5_pro | held_out | 237 | 182 | 166 | 585 |
| wikipedia | gpt_4o | seen | 788 | 609 | 501 | 1898 |

## Split balance

| generator | role | train | dev | test |
|---|---|---|---|---|
| deepseek_v3 | seen | 1186 | 273 | 302 |
| gemini_2_5_pro | held_out | 0 | 280 | 305 |
| gpt_4o | seen | 1304 | 283 | 311 |

Held-out generators appearing in train: **none (correct)**

## Sentence-level label balance

- Total sentences: **39358**
- AI sentences: **14318** (36.4%)
- Human sentences: **25040** (63.6%)

## Boundary-position distribution

Normalised position of the first boundary, by decile. A healthy dataset is spread across deciles; a spike in the first decile means position alone predicts the boundary.

| decile | range | count | share |
|---|---|---|---|
| 0 | 0.0-0.1 | 87 | 2.0% # |
| 1 | 0.1-0.2 | 434 | 10.2% ##### |
| 2 | 0.2-0.3 | 939 | 22.1% ########### |
| 3 | 0.3-0.4 | 780 | 18.4% ######### |
| 4 | 0.4-0.5 | 611 | 14.4% ####### |
| 5 | 0.5-0.6 | 723 | 17.0% ######## |
| 6 | 0.6-0.7 | 331 | 7.8% ### |
| 7 | 0.7-0.8 | 259 | 6.1% ### |
| 8 | 0.8-0.9 | 80 | 1.9%  |
| 9 | 0.9-1.0 | 0 | 0.0%  |

- min=0.083 p25=0.250 p50=0.375 p75=0.500 max=0.833
- First-decile share: 2.0% -> **OK - spread across the passage**

## Span-position distribution (Types 2 and 3)

| decile | count |
|---|---|
| 0 | 0 |
| 1 | 257 |
| 2 | 336 |
| 3 | 573 |
| 4 | 562 |
| 5 | 994 |
| 6 | 638 |
| 7 | 349 |
| 8 | 643 |
| 9 | 0 |

## Retry counts

| retries | records |
|---|---|
| 0 | 3260 |
| 1 | 646 |
| 2 | 322 |
| 3 | 13 |
| 4 | 2 |
| 5 | 1 |

![by generator](boundary_positions_by_generator.png)