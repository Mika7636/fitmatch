/**
 * The three steps of the profile form.
 *
 * Every option list comes from `/api/meta` via props. Nothing here contains a
 * catalogue value: the API validates against vocabularies it measured from the
 * CSVs and answers a 422 naming the valid set, so a hardcoded list would be
 * both wrong and loud about it.
 */

import { useProfile } from '../../context/ProfileContext'
import { limitFor, useMeta } from '../../context/MetaContext'
import { formatPrice } from '../../lib/format'
import { DualRange } from '../ui/DualRange'
import {
  ChoiceField,
  MultiSelectField,
  NumberField,
  SelectField,
} from '../ui/fields'
import { Eyebrow } from '../ui/primitives'

// ---------------------------------------------------------------------------
// Step 1 -- the athlete
// ---------------------------------------------------------------------------
export function StepAthlete() {
  const { draft, setField, errors } = useProfile()
  const { meta, loading } = useMeta()

  const age = limitFor(meta, 'age')
  const height = limitFor(meta, 'height_cm')
  const weight = limitFor(meta, 'weight_kg')
  const shoe = limitFor(meta, 'shoe_size')

  return (
    <div className="space-y-7">
      <div>
        <Eyebrow>Step 1 of 3</Eyebrow>
        <h2 className="mt-2 text-3xl text-ash">The athlete</h2>
        <p className="mt-2 text-sm text-ash-muted">
          Sizing drives the fit constraints. Shoe size is matched within half a
          US size and apparel within one step of yours.
        </p>
      </div>

      <div className="grid gap-6 sm:grid-cols-2">
        <NumberField
          id="age"
          label="Age"
          value={draft.age}
          onChange={(value) => setField('age', value)}
          min={age.min}
          max={age.max}
          step={1}
          unit="yrs"
          error={errors.age}
          placeholder={`${age.min}–${age.max}`}
        />
        <NumberField
          id="shoe_size"
          label="Shoe size (US)"
          value={draft.shoe_size}
          onChange={(value) => setField('shoe_size', value)}
          min={shoe.min}
          max={shoe.max}
          step={meta?.shoe_size.step ?? 0.5}
          error={errors.shoe_size}
          placeholder={`${shoe.min}–${shoe.max}`}
          hint={
            meta
              ? `Catalogue stocks US ${meta.shoe_size.min}–${meta.shoe_size.max}.`
              : undefined
          }
        />
        <NumberField
          id="height_cm"
          label="Height"
          value={draft.height_cm}
          onChange={(value) => setField('height_cm', value)}
          min={height.min}
          max={height.max}
          step={0.1}
          unit="cm"
          error={errors.height_cm}
          placeholder={`${height.min}–${height.max}`}
        />
        <NumberField
          id="weight_kg"
          label="Weight"
          value={draft.weight_kg}
          onChange={(value) => setField('weight_kg', value)}
          min={weight.min}
          max={weight.max}
          step={0.1}
          unit="kg"
          error={errors.weight_kg}
          placeholder={`${weight.min}–${weight.max}`}
        />
      </div>

      <ChoiceField
        label="Gender"
        value={draft.gender}
        onChange={(value) => setField('gender', value)}
        options={meta?.gender ?? []}
        optional
        hint="Filters products by who they are made for; unisex items always qualify. Leave it unset to see everything."
      />

      <div className="grid gap-6 sm:grid-cols-2">
        <ChoiceField
          label="Apparel size"
          value={draft.apparel_size}
          onChange={(value) => setField('apparel_size', value)}
          options={meta?.apparel_size ?? []}
          optional
        />
        <ChoiceField
          label="Foot arch"
          value={draft.foot_arch_type}
          onChange={(value) => setField('foot_arch_type', value)}
          options={meta?.foot_arch_type ?? []}
          optional
          hint="A low arch wants supportive footwear, a high arch wants neutral. Only affects shoes."
        />
      </div>

      {loading && (
        <p className="text-xs text-ash-muted">Loading catalogue options…</p>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Step 2 -- the training
// ---------------------------------------------------------------------------
export function StepTraining() {
  const { draft, setField, errors } = useProfile()
  const { meta } = useMeta()
  const workouts = limitFor(meta, 'workouts_per_week')

  return (
    <div className="space-y-7">
      <div>
        <Eyebrow>Step 2 of 3</Eyebrow>
        <h2 className="mt-2 text-3xl text-ash">The training</h2>
        <p className="mt-2 text-sm text-ash-muted">
          Sport is the strongest signal in the system. Footwear must match it;
          apparel and accessories are treated as sport-flexible.
        </p>
      </div>

      <ChoiceField
        label="Primary sport"
        value={draft.primary_sport}
        onChange={(value) => setField('primary_sport', value)}
        options={meta?.sport_type ?? []}
        hint="Lifestyle products suit any sport, so they stay in the running whatever you pick."
      />

      <div className="grid gap-6 sm:grid-cols-2">
        <ChoiceField
          label="Fitness level"
          value={draft.fitness_level}
          onChange={(value) => setField('fitness_level', value)}
          options={meta?.fitness_level ?? []}
          optional
        />
        <ChoiceField
          label="Train indoors or outdoors"
          value={draft.indoor_or_outdoor}
          onChange={(value) => setField('indoor_or_outdoor', value)}
          options={meta?.indoor_or_outdoor ?? []}
          optional
        />
      </div>

      <div className="grid gap-6 sm:grid-cols-2">
        <NumberField
          id="workouts_per_week"
          label="Workouts per week"
          value={draft.workouts_per_week}
          onChange={(value) => setField('workouts_per_week', value)}
          min={workouts.min}
          max={workouts.max}
          step={1}
          error={errors.workouts_per_week}
          optional
          placeholder={`${workouts.min}–${workouts.max}`}
        />
        <SelectField
          id="climate"
          label="Climate"
          value={draft.climate}
          onChange={(value) => setField('climate', value)}
          options={meta?.climate ?? []}
          optional
          hint="Maps to seasonality: hot and tropical prefer summer gear, cold and continental prefer winter. All-season items always qualify."
        />
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Step 3 -- the preferences
// ---------------------------------------------------------------------------
export function StepPreferences() {
  const { draft, setField, errors } = useProfile()
  const { meta } = useMeta()

  // Slider bounds come from the observed user budgets in users.csv, widened if
  // a loaded sample sits outside them.
  const budgetFloor = meta?.budget.min ?? 10
  const budgetCeiling = meta?.budget.max ?? 450
  // An untouched budget is not a budget: both fields empty means neither is
  // posted, so the hard price constraint is switched off entirely. The slider
  // says "Any price" rather than implying the range it is parked on.
  const budgetUnset = !draft.budget_min.trim() && !draft.budget_max.trim()
  const currentLow = Number(draft.budget_min) || budgetFloor
  const currentHigh = Number(draft.budget_max) || budgetCeiling
  const sliderMin = Math.floor(Math.min(budgetFloor, currentLow))
  const sliderMax = Math.ceil(Math.max(budgetCeiling, currentHigh))

  const setBudget = (low: number, high: number) => {
    setField('budget_min', String(Math.round(low)))
    setField('budget_max', String(Math.round(high)))
  }

  return (
    <div className="space-y-8">
      <div>
        <Eyebrow>Step 3 of 3</Eyebrow>
        <h2 className="mt-2 text-3xl text-ash">The preferences</h2>
        <p className="mt-2 text-sm text-ash-muted">
          Budget is a hard constraint and never bends. Everything else on this
          step is soft — stated as a preference, given up if it leaves too
          little to choose from.
        </p>
      </div>

      <div className="border border-ink-600 bg-ink-800 p-5">
        <DualRange
          label="Budget"
          min={sliderMin}
          max={sliderMax}
          step={5}
          low={currentLow}
          high={currentHigh}
          onChange={setBudget}
          format={(value) => formatPrice(value)}
          unset={budgetUnset}
          unsetLabel="Any price"
          error={errors.budget_max ?? errors.budget_min}
          hint={
            budgetUnset
              ? `Drag either handle to set one. Left alone, no budget is sent at all and every price qualifies${meta ? ` — the catalogue runs ${formatPrice(meta.price.min)} to ${formatPrice(meta.price.max)}` : ''}.`
              : meta
                ? `A hard constraint: nothing outside this range is ever shown. Catalogue prices run ${formatPrice(meta.price.min)} to ${formatPrice(meta.price.max)}.`
                : 'A hard constraint: nothing outside this range is ever shown.'
          }
        />
        <div className="mt-5 grid gap-4 sm:grid-cols-2">
          <NumberField
            id="budget_min"
            label="Minimum"
            value={draft.budget_min}
            onChange={(value) => setField('budget_min', value)}
            min={0}
            step={1}
            unit="$"
            error={errors.budget_min}
          />
          <NumberField
            id="budget_max"
            label="Maximum"
            value={draft.budget_max}
            onChange={(value) => setField('budget_max', value)}
            min={0}
            step={1}
            unit="$"
            error={errors.budget_max}
          />
        </div>
      </div>

      <MultiSelectField
        label="Preferred brands"
        values={draft.preferred_brands}
        onChange={(values) => setField('preferred_brands', values)}
        options={meta?.brand ?? []}
        optional
        hint="Soft: kept if enough products qualify, dropped before size if not."
      />

      <ChoiceField
        label="Style"
        value={draft.style_preference}
        onChange={(value) => setField('style_preference', value)}
        options={meta?.style_preference ?? []}
        optional
        hint="Not a filter — it feeds the TF-IDF ranking through the words it maps onto."
      />

      {/*
        Colour and material are called out as the first things to be given up.
        Session 2 measured colour relaxed for 98.7% of profiles and material
        for 70.2%; saying so here means the relaxation notice on the results
        page confirms something the user was already told.
      */}
      <div className="border-l-2 border-crimson-700 bg-ink-800 p-5">
        <p className="eyebrow text-ash-dim">The first to be given up</p>
        <p className="mt-2 text-xs leading-relaxed text-ash-muted">
          Colour is relaxed for 98.7% of profiles and material for 70.2%. State
          them if you have a preference — you will be told plainly if they could
          not be honoured — but leaving them unset is the cleaner request.
        </p>

        <div className="mt-4 grid gap-6 sm:grid-cols-2">
          <SelectField
            id="color_preference"
            label="Colour"
            value={draft.color_preference}
            onChange={(value) => setField('color_preference', value)}
            options={meta?.color_preference ?? []}
            optional
            placeholder="No colour preference"
          />
          <SelectField
            id="material_preference"
            label="Material"
            value={draft.material_preference}
            onChange={(value) => setField('material_preference', value)}
            options={meta?.material_preference ?? []}
            optional
            placeholder="No material preference"
          />
        </div>
      </div>
    </div>
  )
}
