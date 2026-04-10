# Deploy Code vs. Deploy Model

A consolidated comparison of the two dominant MLOps promotion patterns.

---

## Comparison

| Dimension | Deploy Code | Deploy Model |
|---|---|---|
| **What gets promoted** | Training + feature pipeline code | Serialized model artifact |
| **Where training happens** | Each environment, including prod | Once, in staging |
| **Validated artifact = deployed artifact** | No — re-training in prod yields a slightly different model | Yes — exact artifact is promoted |
| **Training cost** | High — multiple training runs across envs | Low — single training run |
| **Determinism risk** | High — GPU ops, shuffling, etc. can shift model behavior | None |
| **Requires model registry** | Optional | Required (MLflow, SageMaker, etc.) |
| **Staging validation meaningfulness** | Low — metrics on non-prod data are unreliable | High — full validation on prod-quality data |
| **Prod environment role** | Training + serving | Serving only |
| **Rollback strategy** | Re-deploy previous code + re-train | Re-promote previous registered artifact |
| **Operational risk in prod** | Higher — untested training code runs in prod | Lower — only inference runs in prod |
| **Pipeline complexity** | Lower infra, higher operational risk | Higher infra, lower operational risk |

---

## When to Use Each

### Use Deploy Model when

- Staging has access to production-level or production-mirror data
- Model determinism and audit trails are critical (regulated industries, financial, healthcare)
- Training is expensive and you want a single authoritative run
- You need strict "what you validated is what you serve" guarantees

### Use Deploy Code when

- Staging has no access to production data (compliance, legal, privacy constraints)
- The model retrains frequently on fresh prod data (e.g., daily retraining pipelines)
- The team lacks infrastructure for a model registry and artifact store
- Model performance differences between training runs are acceptable and monitored via champion/challenger evaluation in prod

---

## Recommendation

If you can engineer even a sanitized data mirror into staging, **Deploy Model** is the safer, more auditable, and less operationally risky pattern of the two. Deploy Code is the pragmatic fallback when data access is the binding constraint — not the preferred default.

---

## Where This Project Sits

This project uses a **hybrid approach**. Dev and staging use Deploy Code — each
trains its own models using the code deployed to that environment. Staging
trains with production-scale data (100% users, prod hyperparams) so the model
is validated under real conditions. Prod uses Deploy Model — it copies the
`@Champion` artifact from staging rather than training independently. See the
[Model Promotion Guide](model-promotion-guide.md) for the full lifecycle.
