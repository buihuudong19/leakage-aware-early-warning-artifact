# Literature audit (Table 1 of the article)

`literature_audit.csv` codes the evaluation practices of predictive studies that fit models to Olist review or
delivery outcomes. The search covered bibliographic databases, publisher sites and preprint servers up to
October 2026.

## Sets

| IDs | Set | Used in Table 1 |
|---|---|---|
| P01–P17 | Peer-reviewed papers and preprints; Olist named (12), or not named but probably Olist (4) or unconfirmed (1) | yes |
| U01–U02 | Peer-reviewed papers whose data could not be matched to Olist | no |
| T01–T05 | Graduate theses (grey literature) | no |

## Fields

| Field | Content |
|---|---|
| `id`, `authors`, `year`, `title`, `venue`, `venue_type`, `doi_or_url` | Bibliographic record |
| `verified` | The record was found at the publisher or indexing service |
| `task` | Predicted outcome (binary low rating, multiclass score, satisfaction, late delivery, ...) |
| `prediction_time` | When the prediction is made, as stated by the authors, or `not stated` |
| `split` | `random_holdout`, `temporal`, `not_stated` or `unclear` |
| `post_outcome_features` | Inputs recorded after delivery or review (`yes`, `no`, `unclear`) |
| `full_period_aggregates` | Target aggregates (e.g. seller or product rating means) computed over the full period |
| `label_availability_purge` | Training orders whose label would be unknown at prediction time are removed |
| `class_imbalance_handling`, `best_model_claimed`, `headline_metrics` | As reported |
| `note` | Evidence level (`full text`, `abstract`, `metadata`), how the data set was matched to Olist, and remarks |

## Coding rules

* One person coded all studies. Five studies (P01, P02, P04, P06, P10) were coded from the full text, eleven from the
  abstract and one (P17) from metadata only. P10 was first coded from its abstract and recoded from the full text
  in a second pass.
* A practice that is not reported is coded `not stated` or `unclear`; it is never coded as present. Table 1 therefore
  measures reporting practice, not what the authors did.
* `post_outcome_features = yes` describes the inputs. Whether they are leaky depends on the prediction time, which only
  three of the seventeen studies state.
