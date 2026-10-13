import test from "node:test";
import assert from "node:assert/strict";
import { normalizeCanonicalToolArguments } from "../lib/canonical-tool-arguments.mjs";

test("omits empty and redundant self-beneficiary fields", () => {
  assert.deepEqual(normalizeCanonicalToolArguments("record_medicare_permissions", {
    beneficiary_name: "Patricia Lee", caller_name: "Patricia Lee", caller_relationship: "self",
    callback_phone: "2125550133", consent_statement: "yes", contact_consent: true,
  }), {
    caller_name: "Patricia Lee", caller_relationship: "self", callback_phone: "2125550133",
    consent_statement: "yes", contact_consent: true,
  });
});

test("keeps only caregiver-safe fields for an absent beneficiary", () => {
  assert.deepEqual(normalizeCanonicalToolArguments("save_medicare_qualification", {
    beneficiary_name: "Rosa Rivera", caller_name: "Daniel Rivera", caller_relationship: "adult_child",
    callback_phone: "6025550166", consent_id: "perm_1004", intake_intent: "caregiver_inquiry",
    product_interest: "medicare_advantage", age_band: "unknown", current_coverage: "unknown",
    election_window: "not_sure", state: "", zip_code: "", notes: "",
  }), {
    beneficiary_name: "Rosa Rivera", caller_name: "Daniel Rivera", caller_relationship: "adult_child",
    callback_phone: "6025550166", consent_id: "perm_1004", intake_intent: "caregiver_inquiry",
    product_interest: "medicare_advantage",
  });
});

test("removes sales defaults from a member-services route", () => {
  assert.deepEqual(normalizeCanonicalToolArguments("route_medicare_call", {
    route_reason: "member_services", service_issue_type: "claim", availability_context: "unknown",
    product_interest: "unknown", lead_id: "", state: "", zip_code: "",
  }), { route_reason: "member_services", service_issue_type: "claim" });
});

test("preserves meaningful ordinary-sales arguments", () => {
  const argumentsObject = {
    availability_context: "normal", lead_id: "lead_2001", product_interest: "medicare_advantage",
    route_reason: "sales_or_plan_review", service_issue_type: "not_applicable", state: "CA", zip_code: "94105",
  };
  assert.deepEqual(normalizeCanonicalToolArguments("route_medicare_call", argumentsObject), argumentsObject);
});
