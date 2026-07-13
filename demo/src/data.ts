// ---------------------------------------------------------------------------
// REAL reproduction data. Everything here was pulled from our own runs on the
// ACE paper's FiNER benchmark (XBRL GAAP-tagging), DeepSeek-V3.1 via AWS Bedrock,
// train=1000 / val=500 / test=441.
//
//   - Baseline + GEPA test accuracy and the GEPA instruction trajectory come
//     from experiments/results/finer_fullrun.log (measured).
//   - The ACE playbook is the real artifact ACE authored on this task.
//   - ACE's full-run TEST accuracy is still being measured; set it below when
//     the reproduction lands and the ACE bar goes from "in progress" to solid.
// ---------------------------------------------------------------------------

export const BENCHMARK = {
  name: "FiNER",
  subtitle: "XBRL / US-GAAP tag classification",
  model: "DeepSeek-V3.1",
  provider: "AWS Bedrock (us-west-2)",
  sizes: { train: 1000, val: 500, test: 441 },
} as const;

export type Method = "baseline" | "gepa" | "ace";

// Measured test accuracy (fraction of comma-aligned GAAP tags correct).
// Set `ace` to a number once the full reproduction completes.
export const TEST_ACC: Record<Method, number | null> = {
  baseline: 0.652,
  gepa: 0.707,
  ace: null, // full run in progress — see repo README
};

// A single representative FiNER task (abbreviated from the real test set) that
// streams through all three programs during "training".
export const TASK_PROMPT =
  'You are given US-GAAP tag options and 4 questions, each asking for the best ' +
  'tag for a numerical entity in a financial-disclosure sentence. Answer with a ' +
  'comma-separated list of tags, in order, and nothing else.';

// Real example rows (entity + gold tag) drawn from finer_test_subset_006.
export interface Example {
  id: number;
  entity: string;
  sentence: string;
  gold: string;
}

export const EXAMPLES: Example[] = [
  {
    id: 0,
    entity: "$34.2 million",
    sentence:
      "$34.2 million of unrecognized compensation cost related to unvested RSUs is expected to be recognized over 2.63 years.",
    gold: "EmployeeServiceShareBasedCompensationNonvestedAwardsTotalCompensationCostNotYetRecognized",
  },
  {
    id: 1,
    entity: "$2.87",
    sentence:
      "…the Rights to purchase 8,778,230 shares of common stock, at an exercise price of $2.87 per share…",
    gold: "ClassOfWarrantOrRightExercisePriceOfWarrantsOrRights1",
  },
  {
    id: 2,
    entity: "8,778,230",
    sentence:
      "…the Rights to purchase an aggregate of 8,778,230 shares of common stock…",
    gold: "SaleOfStockNumberOfSharesIssuedInTransaction",
  },
  {
    id: 3,
    entity: "5.7",
    sentence:
      "…the Company recorded a $5.7 million loss upon extinguishment of debt…",
    gold: "GainsLossesOnExtinguishmentOfDebt",
  },
  {
    id: 4,
    entity: "400,000,000",
    sentence:
      "…authorized to issue 400,000,000 shares of common stock, par value $0.001…",
    gold: "CommonStockSharesAuthorized",
  },
  {
    id: 5,
    entity: "ten years",
    sentence:
      "The term of options granted under the 2011 Plan is ten years except for certain grants.",
    gold: "SharebasedCompensationArrangementBySharebasedPaymentAwardExpirationPeriod",
  },
];

// -- GEPA: the real instruction-evolution trajectory (accepted Pareto steps) --
// From finer_fullrun.log. Each accepted candidate improved the val score.
export interface GepaStep {
  iter: number;
  instruction: string;
  val: number; // validation accuracy after this candidate
  accepted: boolean;
}

export const GEPA_TRAJECTORY: GepaStep[] = [
  {
    iter: 0,
    instruction: "Given the field `question`, produce the field `answer`.",
    val: 0.663,
    accepted: true,
  },
  {
    iter: 1,
    instruction:
      "You are an expert in XBRL and US GAAP taxonomy. Answer 4 independent questions by selecting the most appropriate US GAAP tag from the provided list for each numerical entity mentioned in the context sentences.",
    val: 0.708,
    accepted: true,
  },
  {
    iter: 3,
    instruction:
      "You are an expert in XBRL and US GAAP financial reporting. Analyze a list of US GAAP tags and a series of questions, then provide the single most accurate tag for each question.",
    val: 0.717,
    accepted: true,
  },
  {
    iter: 4,
    instruction:
      "You are an expert in US GAAP and XBRL tagging. Analyze a sentence containing a specific numerical or textual entity and select the single most appropriate US GAAP taxonomy tag for it from a provided list.",
    val: 0.725,
    accepted: true,
  },
  {
    iter: 9,
    instruction:
      "You are an expert in US GAAP and XBRL tagging. Your primary task is to select the single most appropriate US GAAP taxonomy tag for a specified entity within a given sentence.",
    val: 0.733,
    accepted: true,
  },
];

// -- ACE: the real playbook it authored on this task --------------------------
export interface Bullet {
  id: string;
  section: string;
  content: string;
}

export const ACE_PLAYBOOK: Bullet[] = [
  {
    id: "of-00007",
    section: "OUTPUT_FORMAT",
    content:
      "Always output only a comma-separated list of tags in the order of the input questions, with no extra text.",
  },
  {
    id: "mg-00005",
    section: "MAPPING_GUIDELINES",
    content:
      "First identify the specific attribute (par value, shares outstanding, shares authorized, shares available under plan) and select the most precise tag available.",
  },
  {
    id: "ts-00006",
    section: "TERMINOLOGY_SENSITIVITY",
    content:
      'Pay attention to exact terms ("par value," "outstanding," "available under plan," "maximum authorized") — they dictate the correct tag.',
  },
  {
    id: "ec-00009",
    section: "EQUITY_COMPENSATION",
    content:
      "For RSU grant counts, use ShareBasedCompensationArrangementByShareBasedPaymentAwardEquityInstrumentsOtherThanOptionsGrantsInPeriod.",
  },
  {
    id: "ec-00010",
    section: "EQUITY_COMPENSATION",
    content:
      "For total unrecognized stock-based comp expense, use EmployeeServiceShareBasedCompensationNonvestedAwardsTotalCompensationCostNotYetRecognized.",
  },
  {
    id: "ec-00008",
    section: "EQUITY_COMPENSATION",
    content:
      "For intrinsic value of options exercised, use ShareBasedCompensationArrangementByShareBasedPaymentAwardOptionsExercisesInPeriodTotalIntrinsicValue.",
  },
  {
    id: "psi-00002",
    section: "PREFERRED_STOCK_ISSUANCE",
    content:
      "Use specific new-share-issued tags (e.g. StockIssuedDuringPeriodSharesNewIssues) instead of general sale tags.",
  },
  {
    id: "cpp-00003",
    section: "CONVERTIBLE_PREFERRED_PRICING",
    content:
      "For convertible preferred issuance price per share, use conversion-term tags (e.g. DebtInstrumentConvertibleConversionPrice1).",
  },
  {
    id: "ira-00004",
    section: "INSURANCE_RESERVE_ACCOUNTING",
    content:
      "For prior-year reserve development, use SupplementalInformationForPropertyCasualtyInsuranceUnderwritersPriorYearClaimsAndClaimsAdjustmentExpense.",
  },
  {
    id: "ec-00001",
    section: "EQUITY_COMPENSATION",
    content:
      "For RSU grants (performance or service-based), use equity-instruments-other-than-options tags.",
  },
];
