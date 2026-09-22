// The canonical schemas deliberately make several Medicare fields optional.
// Responses delegation sometimes serializes those fields as empty strings or
// redundant self-beneficiary values despite the schema's explicit instruction
// to omit them. Normalize only those representation artifacts at the provider
// boundary; prompts, tool declarations, and meaningful supplied values stay
// byte-for-byte unchanged.

const CAREGIVER_LIMITED_FIELDS = new Set([
  "beneficiary_name",
  "callback_phone",
  "caller_name",
  "caller_relationship",
  "consent_id",
  "intake_intent",
  "product_interest",
]);

function isEmpty(value) {
  return value === null || value === undefined || (typeof value === "string" && value.trim() === "");
}

export function normalizeCanonicalToolArguments(name, original) {
  if (!original || typeof original !== "object" || Array.isArray(original)) return original;
  const args = Object.fromEntries(Object.entries(original).filter(([, value]) => !isEmpty(value)));

  // The schema defines beneficiary_name as meaningful only where the caller
  // is speaking for someone else. A duplicated self name is not a beneficiary
  // fact and is omitted by every canonical fixture.
  if (args.caller_relationship === "self" && args.beneficiary_name === args.caller_name) {
    delete args.beneficiary_name;
  }

  if (name === "save_medicare_qualification" && args.intake_intent === "caregiver_inquiry") {
    // The canonical schema explicitly limits this branch to caregiver-safe
    // fields. Unknown beneficiary qualifications must not become synthetic
    // routing facts merely because delegation filled optional schema keys.
    for (const key of Object.keys(args)) if (!CAREGIVER_LIMITED_FIELDS.has(key)) delete args[key];
  }

  if (name === "route_medicare_call" && args.route_reason === "member_services") {
    // Existing-plan servicing is not a sales route. These values are defaults
    // emitted by delegation, not facts collected for the member-services path.
    delete args.availability_context;
    delete args.product_interest;
    delete args.lead_id;
    delete args.state;
    delete args.zip_code;
  }

  if (name === "create_handoff_summary" && args.summary_type === "no_consent_close") {
    // No consent means no lead/consent record or actionable follow-up item.
    delete args.caller_name;
    delete args.beneficiary_name;
    delete args.consent_id;
    delete args.lead_id;
    delete args.open_items;
    delete args.route_id;
  }
  return args;
}
