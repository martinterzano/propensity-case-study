# Donor Propensity to Second Gift: Case Study

> Binary classification model that scored one-time NGO donors by their probability
> of making a second donation, built to feed a reengagement campaign. One of two
> branches that came out of a single scoping decision for the client. This branch
> was validated and recommended against production because the model's lift at any
> operational cutoff did not justify the campaign cost. The recommendation was
> accepted by the client. The other branch of the same scoping (recurring donors)
> became a churn model that did reach production.

**Sector:** non-profit · **Role:** contributor (~35% ownership, team of 2 DS) · **Stack:** Python, scikit-learn, GCP (BigQuery) · **Status:** validated, not advanced to production (DS recommendation, accepted by client)

The client owns the exact business metrics. Performance numbers below are
expressed as directional patterns or small ranges. The population scoping
decisions, the reasons we recommended against production, and the operational
logic are the point of this case study, not the raw metric values.

---

## Contents

1. [The problem](#1-the-problem)
2. [Scoping: two generative processes, two projects](#2-scoping-two-generative-processes-two-projects)
3. [Model and features for the one-time donors branch](#3-model-and-features-for-the-one-time-donors-branch)
4. [Lift by percentile as a go/no-go criterion](#4-lift-by-percentile-as-a-gono-go-criterion)
5. [Why we recommended against production](#5-why-we-recommended-against-production)
6. [Related work: the recurring donors branch](#6-related-work-the-recurring-donors-branch)
7. [Engineering architecture and shared pipeline](#7-engineering-architecture-and-shared-pipeline)
8. [Scoping decisions and repository contents](#8-scoping-decisions-and-repository-contents)
9. [What this case study is meant to show](#9-what-this-case-study-is-meant-to-show)

---

## 1. The problem

An NGO client was running reengagement campaigns on its donor base. Targeted
outreach to lapsed and inactive donors, aimed at prompting a second donation.
Across all contacted donors the campaigns were reaching a baseline reengagement
rate around 18%.

The question the model was designed to answer. Among the current pool of
one-time donors (those who had made exactly one donation), which ones have the
highest probability of making a second gift if contacted?

The expected deliverable was a ranked list of one-time donors, scored by their
predicted probability of a second donation, used by the marketing team to
prioritise outreach within their weekly campaign capacity.

The underlying assumption at the start of the project was that a ranking model
would let the team concentrate the fixed campaign budget on the segment with the
highest conversion rate, lifting the overall campaign ROI above the 18%
baseline. Whether that assumption held is what §4 and §5 are about.

## 2. Scoping: two generative processes, two projects

Before any model was trained, a specific subgroup was identified in the dataset.
Donors who, at the time of their first donation, had also committed as recurring
donors (scheduled monthly givers).

### 2.1 Why this mattered more than it looked

Recurring donors showed a second-gift rate well above the one-time donor
population. On the surface this looked like a positive signal. In practice it
was the opposite. The high rate was mechanical, not behavioural. Recurring
donors had committed to a monthly pledge at the time of their first donation,
funded in most cases by an automated monthly charge, and the median donor
stayed on the pledge for more than ten months. Their "second gift" was
essentially the next scheduled charge against an active subscription, not a
response to campaign messaging.

Mixing them into the training set would have inflated the apparent baseline
conversion rate and painted a misleading picture of what a reengagement
campaign could achieve. A model trained on the pooled population would learn
that "second gift" was a near-certain event for a sizable subgroup, and the
lift metrics would look strong for the wrong reason. The marketing team would
then expect the campaign to produce conversion rates it was never going to
reach on the actual target segment. One-time donors who have no active
commitment and whose next gift depends entirely on external stimulus.

**One-time donors making a voluntary second gift** are responding to an external
stimulus. A reengagement campaign, cause salience, a direct ask, a positive
previous experience. The features that predict their behaviour are attributes of
the donor and attributes of the contact strategy.

**Recurring donors fulfilling or breaking a prior monthly pledge** are
responding to internal circumstances. Payment capacity, a change of mind about
the pledge, life events. The features that predict whether they continue are
attributes of the commitment structure, not of campaign messaging.

Training a single model on both populations would have produced a system that
learned to discriminate across two different causal stories at the same time.
The resulting predictions would be incoherent. Some donors would be scored high
because their profile resembled "recurring donors who continue"; others because
their profile resembled "one-time donors who respond to outreach". The marketing
team would not have been able to act on that ranking coherently, because the
retention action for a wavering recurring donor and the acquisition action for
a reengagement-eligible one-time donor are different operational workflows run
by different teams.

There was also a practical scope issue. Recurring donors with an active pledge
are not a relevant target for a reengagement campaign in the first place.
Contacting a donor who already has a monthly commitment active with a
second-gift request is a workflow collision, not an opportunity.

### 2.2 The alternative considered and rejected

The alternative was to include recurring donors in the training set and add a
binary feature (`is_recurring_donor`) so the model could learn to adjust
predictions based on that indicator.

Rejected for two reasons. First, it made the model's reliability in production
dependent on the quality of that flag's labeling. If the flag was mislabeled or
missing for some records, the model would fail silently, producing confidently
wrong predictions for the affected donors without any visible error signal.
Second, explicit exclusion is more transparent than implicit adjustment. The
model's scope is documented clearly in the problem definition, not buried inside
model logic.

### 2.3 How a scoping decision bifurcated the engagement into two projects

Once the two populations were separated at the data level, the engagement with
the client split into two parallel questions.

- **One-time donors.** Does a ranking model on this segment lift the reengagement campaign enough to justify production? This is the project this case study covers.
- **Recurring donors.** Can we predict which active recurring donors are about to miss their next monthly gift, so retention outreach can be targeted? This became a separate churn project (see §6).

Two populations, two problems, two models, two deliverables. The decision to
split was the natural consequence of recognising
that two different causal stories were present in what had initially been
scoped as one project.

## 3. Model and features for the one-time donors branch

Training set. Approximately 50,000 one-time donor records, drawn from donations
made over a two-year period.

### Feature groups available

| Group | Examples |
|---|---|
| Transactional | First donation amount, first donation date, donation channel |
| Demographic | Age bracket, geographic region |
| Timing within the year | Month, quarter, weekday of first donation |

The feature set was deliberately narrow because the available data was narrow.
The client's donor system captured transactional and basic demographic
attributes with a lot of missing data on demographics which made it impossible to use. Behavioural engagement data (email opens, campaign responses outside
of donation, cause-area affinity signals) was not available at the time of the
project. Whether that limitation ended up being the binding constraint is what
§5 is about.

### Model choice and class balance

**Random Forest classifier.** Tree-based, handles mixed numeric and categorical
features without heavy preprocessing, produces calibrated probabilities (with
`CalibratedClassifierCV` wrapping if needed), and interacts features naturally.

**Class balance.** The positive class (donors who made a second gift) was around
18% of the dataset. Moderate imbalance. Manageable without aggressive
resampling. `class_weight="balanced"` on the RF was enough. SMOTE and
undersampling were tried in exploratory experiments and did not change the
ranking meaningfully.

**Validation.** Multiple train/validation splits were run, including a time-split
variant (train on earlier donations, validate on later ones) and cohort-based
splits by first-donation period. The relevant finding for this case study came
from both; see §4 and §5.

The exact feature list, hyperparameters, and feature-importance ranking are
client property.

## 4. Lift by percentile as a go/no-go criterion

The model outputs a continuous probability per donor. That probability by itself
does not define a business decision. The marketing team's operational interface
was going to be a ranked list cut at some depth. Contact the top X% this week.

The lift-by-percentile analysis is the standard tool to decide where to cut. It
also happens to be the tool that answers the question *before* the cut. Is
there a cut that justifies running the campaign at all?

### 4.1 How lift is read for a go decision

The probability distribution is divided into deciles and percentiles. For each decile or percentile, the
observed second-gift rate on the validation set is compared against the 18%
population base rate. For a model with strong signal, the top decile or percentile shows a
second-gift rate well above 18%, the second decile shows a smaller but still
positive excess, rates decline progressively, and somewhere down the list the
rate crosses the base rate.

Combining this with campaign economics (send cost, management cost, expected
donation value, donor fatigue) gives a per-decile ROI estimate. The classical
output is a cutoff. The percentile below which campaign activation produces
negative expected ROI. The recommendation becomes "activate the top K%".

### 4.2 How the same analysis produces a no-go

A lift curve can also answer a different question. Given the campaign economics,
is there any percentile at which activation is clearly positive? If the top
decile's second-gift rate is only marginally above the base rate, and the
difference shrinks fast as the model goes deeper, the per-decile ROI may be
marginal at the top of the list and negative everywhere else. In that case the
tool does not output a cutoff. It outputs a verdict. No cutoff is worth running.

This is where this project landed. The same decile-level lift computation
that would normally produce "activate the top K%" produced "do not activate
at any K" once the per-decile ROI was factored in, and that output was
treated as a first-class result, not a null-return edge case.

### 4.3 Why lift by percentile and not a fixed probability threshold

A fixed probability threshold ("contact everyone above 0.40") has two problems
that make it worse than useless at the go/no-go stage.

First, the right threshold depends on campaign economics, not on the model's
internal probability scale. The threshold that maximises ROI in a low-cost
email campaign is different from the threshold that maximises ROI in a phone
campaign for the same model, even though the model has not changed.

Second, if the model is retrained or recalibrated, probabilities shift and the
threshold loses its meaning. The threshold has to be recalibrated as well.

Lift by percentile sidesteps both problems. Decile ranks are inherently
self-normalising. The top decile is always the top 10% of the current ranked
list, regardless of how the score distribution has shifted. And the economic
threshold (which percentile is worth activating) is computed separately from
the model, so it can be recalibrated when campaign costs or expected donation
value change without touching the model.

If the NGO were to launch a matching-donations program that doubles the
expected value of each second gift, the lift analysis can be re-run on the same
model to find a new activation cutoff. The model does not need to be retrained.
This flexibility is what the operational interface buys, and it holds whether
the final decision is "activate the top 30%" or "do not activate anywhere".

## 5. Why we recommended against production

The lift analysis described in §4 produced a no-go verdict. This part of the
case study describes what that looked like in practice, what the alternative
would have required, and why "deploy anyway and see" was not an acceptable
fallback.

### 5.1 What "signal insufficient" meant operationally

The top decile's second-gift rate was above the 18% baseline, and the lift was
positive. The size of the lift, combined with the campaign cost per contact and
the expected donation value, did not produce positive ROI once the per-contact
cost was subtracted. Below the top decile the lift collapsed quickly toward 1.0
(no better than random), so there was no deeper band that was cost-effective
either.

Several variations of the model were tried. With and without
`scale_pos_weight`, with and without the `is_recurring_donor` population
included (as a diagnostic), on random splits and on time-based splits, and on
cohorts of donors from different two-year windows. The ranking produced by the
best version of each variant looked structurally similar. The top decile
separated from the rest, and not by enough. No modelling choice shifted the
economics from "no" to "yes".

The model was learning something real. It was just not learning enough.

### 5.2 The alternative that was not available

The feature set in §3 captured what the donor had done at the point of the
first donation (amount, channel, date, demographics). It did not capture what
the donor had done since then. A one-time donor who had opened six campaign
emails over the past year, clicked through on two, and spent time on the cause
page is behaviourally different from one who had engaged with none of the above.
Both look identical in the feature set used.

Adding behavioural engagement features (email open and click history, visit
patterns, campaign exposure counts) is the standard next step for a propensity
model of this kind. For the client, this data existed inside the marketing
automation platform and was not integrated with the donor system at the time of
the project. Integrating it was a scoped-out piece of work with its own cost
and timeline.

[TAMBIEN HAY QUE EXPLICAR, COMO DIJE ARRIBA, QUE LA DATA DE DEMOGRAPHICS ESTABA INCOMPLETA EN UN ALTO PORCENTAJE Y QUE NO FUE POSIBLE USARLO. Y QUE, EN GENERAL, UN MODELO DE PROPENSION QUE SOLO USA DATA TRANSACCIONES Y CUENTA QUE MUY POCA DATA DISPONIBLE TIENDE A NO PRODUCIR RESULTADOS DECENTES AUNQUE POR SUPUESTO HAY EXCEPCIONES QUE HACEN QUE VALGA LA PENA EXPERIMENTAR]

### 5.3 Why "deploy anyway and see" was rejected

A tempting fallback at this stage is to deploy the model with the available
signal, measure incrementality through an A/B test against the uniform-outreach
baseline, and improve from there. We discussed it with the client and
recommended against it.

The economic cost of running the campaign on the model-selected top decile for
long enough to measure incrementality (and the opportunity cost of not running
the uniform campaign that was already producing the 18% baseline) was
non-trivial. More importantly, the result was predictable from the lift
analysis. The model's top decile was not separated enough from the baseline for
an A/B test on the achievable sample size to detect a difference with
reasonable power. The test would most likely come back inconclusive, and
inconclusive is the worst outcome. It would consume budget without resolving
the deployment question, and would also delay the alternative work (the
recurring donors branch, see §6) by months.

The recommendation was. Do not productionize this model. Redirect the
client-side investment toward integrating behavioural engagement data [AN IMPROVING DEMOGRAPHICS DATA] (which
would make a future version viable) and move the second-population branch
forward in parallel. The client accepted the recommendation.

### 5.4 What the recommendation did and did not say

The recommendation was not that donor propensity modeling for second gifts is
infeasible in general. It was that this specific model, on this specific data,
with this specific feature set, did not justify production at the time of the
project. Three things could change that verdict in a future iteration.

- **Better features.** Behavioural [AND DEMOGRAPHICS] engagement data would almost certainly produce a stronger ranking. The feature groups that typically carry signal in propensity models of this kind (recent interaction patterns, cause-area affinity, campaign exposure) were systematically absent here.
- **A lower-cost outreach channel.** The economics are a function of cost per contact. A sufficiently cheap channel (triggered email with no manual follow-up) could make the current lift pay even without new features.
- **A different donation-side target.** The current label is "second gift of any size within the horizon". A model targeting a higher-value second gift, with different per-contact economics, is a different problem and would require re-scoping.

The recommendation documented these three paths, so that the decision to not
deploy was framed as "not this version" rather than "not this problem".

## 6. Related work: the recurring donors branch

The population excluded in §2 was addressed as a separate project. The two
projects shared the client, the overall engagement, and parts of the data
engineering pipeline. They did not share the modelling problem.

Recurring donors give at a near-monthly cadence. Churn is binary and clean. Did
the next monthly gift arrive within the expected window, or not? Features that
had no traction on the one-time problem (donor's own historical cadence,
recency relative to that cadence, gap patterns) carry strong signal here. The
operational action is retention outreach, which has different per-contact
economics than acquisition outreach, and the segmentation is driven by a
combination of recency bands and model percentile within those bands.

The churn project did reach production, with a monthly scoring cadence feeding
a prioritised retention workflow.

Case study: [churn-donors-case-study](https://github.com/martinterzano/churn-donors-case-study).

## 7. Engineering architecture and shared pipeline

The model this case study describes is one output of a project. The other
output, which outlived the model, is a modular training pipeline built as
reusable infrastructure from the first iteration. When the recurring donors
branch (§6) started, the pipeline was already there. Churn-specific feature
engineering and configuration were layered on top. The shared pipeline is
what let the second project move from blank slate to full validation grid in
a fraction of the time it would have taken to rebuild.

### 7.1 Modular sklearn Pipeline

The training pipeline is composed as a sklearn `Pipeline` with five stages,
each a separate module.

```
Pipeline([
    ('mapper',           FeatureMapper()),      # client-specific categorical mapping
    ('feature_engineer', FeatureEngineer()),    # derived features (log, bucket, interaction)
    ('outlier_clipper',  OutlierClipper()),     # per-column quantile clipping, fitted on train
    ('preprocess',       build_preprocessor()), # impute + scale + one-hot
    ('model',            get_model(name)),      # LR / RF / XGBoost, swapped by name
])
```

Each stage is a `BaseEstimator` + `TransformerMixin` subclass with a plain
`fit` / `transform` contract, which gives three things for free. `GridSearchCV`
can search over any stage's hyperparameters without caring about the others.
`joblib.dump` serialises the whole pipeline (including fitted state) in one
call. And `cross_val_score` runs end-to-end without data leakage, because
`OutlierClipper` and `FeatureEngineer` fit on the training fold only and
transform the validation fold without re-fitting.

### 7.2 Why the modules are separated the way they are

The split follows the principle that each module should do one thing with a
different kind of state.

| Module | What it owns | Why it is separate |
|---|---|---|
| `FeatureMapper` | Client-specific categorical mappings (payment method normalisation, engagement group taxonomy, campaign-name grouping) | Business taxonomy changes without the model changing. Keeping it in its own module makes "the client renamed a category" a one-file edit. |
| `FeatureEngineer` | Pure derived features: logs, buckets, interactions, date-part extraction | Stateless by design. No `fit` state. Separable because the derived-features catalogue grows faster than the rest of the pipeline. |
| `OutlierClipper` | Per-column quantile bounds, fitted on training data | Has fit state (`bounds_`), applies at transform time with `clip`. Separated so the clipping policy is auditable without opening the preprocessor. |
| `build_preprocessor` | Numeric: `IterativeImputer` + `StandardScaler`. Categorical: `SimpleImputer` + `OneHotEncoder(handle_unknown='ignore')`. | A function returning a `ColumnTransformer`, not a module. The behaviour is standard sklearn; wrapping it in a builder keeps the main `pipeline.py` thin. |
| `get_model` | A single switch returning an unfitted classifier by name, with `class_weight="balanced"` wired in | Lets the experiment grid loop over `['logistic', 'random_forest', 'xgboost']` without touching the pipeline composition. The imbalance treatment is defined at this boundary, not scattered through the training notebook. |

The practical consequence is that a new experiment is a function call, not an
engineering effort. Switching from `random_forest` to `xgboost` is one string.
Changing the clipping range from the default 5–95 percentiles to 1–99 is one
argument. Adding a new derived feature is a method on `FeatureEngineer` and
every downstream experiment picks it up automatically.

### 7.3 Experiment logging to BigQuery

Every training run writes a row to a dedicated BigQuery table with the
run's parameters, metrics on train and test, cross-validated recall, and the
overfit gap (train AUC minus test AUC). Logging happens inside the training
helper, so forgetting to log a run would require actively bypassing it.

```
run_and_log_experiment(model, params, X_train, y_train, X_test, y_test, model_name, description='', table_id=...)
```

This single-run logger was later evolved in the churn project into a
two-table structure (one row per run + one row per fold) to accommodate
the walk-forward validation loop.

### 7.4 What this architecture bought downstream

- The recurring donors project (§6) reused the full stack (`pipeline.py`, `preprocessor.py`, `outlier_clipper.py`, `model_factory.py`) and only added its own `FeatureEngineer` subclass and its own config file. First end-to-end training run on the recurring population was days, not weeks.
- The experiment logger pattern was expanded in the churn project into a two-table structure (runs + folds) to accommodate the walk-forward validation loop, with the propensity-era single-run logger as the direct ancestor.
- When new DS joined the engagement, there was documented infrastructure to extend. Reading the `pipeline.py` composition answered "where does X happen?" in one place.

The code artefacts outlived the production decision. The model this case
study describes did not reach production (§5). The pipeline it was built on
is the thing that is still in use.

## 8. Scoping decisions and repository contents

### What is deliberately not published

- **Client identity, campaign-level metrics, exact feature list, exact feature-importance ranking.** Client property.
- **Model artefacts, training SQL, scoring SQL.** Client property.
- **Hyperparameter grids, exact validation split dates, exact per-decile numbers.** Client property. The decisions described in §4 and §5 are based on these numbers; the shape and direction are described, the raw values are not.

### What is in this repo

```
case-propensity/
├── README.md    ← this document
└── AUDIT.md     ← pre-publish checklist
```

The full training code (notebooks, feature engineering modules, scoring
scripts, and the shared pipeline modules described in §7) is the client's
property and is not published here. The architecture is documented in §7
with the module boundaries, state ownership, and composition pattern spelled
out, which is enough to see the shape of the engineering without exposing
client logic.

## 9. What this case study is meant to show

- **Population scoping as a domain decision, not a data decision.** The most important work in this project happened before a single model was trained. Recognising that two different causal stories were present in the data, and that the right response was two models for two populations, not one model with an adjustment flag.
- **Two distinct generative processes warrant two distinct models.** Mixing one-time donors and recurring donors into a single training set would have produced a system answering two questions at once. The resulting ranking would not have been operationally actionable, because the retention and acquisition workflows live in different teams with different actions.
- **Explicit exclusion over feature-based adjustment.** When a flag variable is required to handle a subpopulation correctly, the implicit cost is a dependency on labeling quality in production. Explicit exclusion removes that dependency at the cost of scope. A trade-off worth making when the excluded segment falls outside the campaign's intended audience.
- **Lift by percentile as a go/no-go filter, not just a cutoff tool.** The decile analysis has two outputs that look like one. "Where to cut" is the obvious one. "Whether to cut at all" is the one that matters when the lift is marginal. Collapsing both into a single number ("the recommended threshold is X") hides the second output and makes the no-go verdict harder to see.
- **Recommending against production as a senior deliverable.** A validated model that does not justify its operational cost is a result, not a failure. The recommendation to not deploy was documented alongside the three paths that would change the verdict (better features, cheaper channel, higher-value target), so the decision was framed as "not this version" rather than "not this problem".
- **A scoping decision can bifurcate an engagement into two projects with different outcomes.** This is not a pivot. It is the natural consequence of recognising that what was scoped as one problem is really two. One of the two branches can reach production and the other can be shelved, and both outcomes are coherent with the original scoping decision.
- **Infrastructure built once, reused across projects.** The modular sklearn Pipeline described in §7 was built as reusable scaffolding from the first iteration of this project. When the recurring donors branch started, the pipeline was already there and the second project moved from blank slate to full validation grid in days, not weeks. The model this case study describes did not reach production, and the pipeline it was built on is the thing that is still in use.

---

Martín Terzano · [helliumlab.com](https://helliumlab.com) · [LinkedIn](https://www.linkedin.com/in/martinterzano) · martin@helliumlab.com
