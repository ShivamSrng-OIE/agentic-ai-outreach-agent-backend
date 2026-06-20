"""Domain enums."""

from enum import StrEnum


class SourceField(StrEnum):
    COMPANY_NAME = "company_name"
    COMPANY_DESCRIPTION = "company_description"
    CULTURE_AND_VALUES = "culture_and_values"
    HIRING_PROFILES = "hiring_profiles"
    COMMUNICATION_TONE = "communication_tone"
    RECRUITING_INTENT = "recruiting_intent"
    ADDITIONAL_CONTEXT = "additional_context"


class EvidenceKind(StrEnum):
    COMPANY_IDENTITY = "company_identity"
    PRODUCT = "product"
    CULTURE = "culture"
    WORKING_STYLE = "working_style"
    HIRING_PROFILE = "hiring_profile"
    ROLE_INFORMATION = "role_information"
    COMPENSATION = "compensation"
    VISA_OR_SPONSORSHIP = "visa_or_sponsorship"
    LOCATION = "location"
    WORK_MODE = "work_mode"
    BENEFITS = "benefits"
    FUNDING = "funding"
    COMMUNICATION_GUIDANCE = "communication_guidance"
    RECRUITING_INSTRUCTION = "recruiting_instruction"


class CandidateIntent(StrEnum):
    INTERESTED = "interested"
    ASKS_FOR_MORE_INFORMATION = "asks_for_more_information"
    ASKS_WHY_CONTACTED = "asks_why_contacted"
    ASKS_ABOUT_ROLE = "asks_about_role"
    ASKS_ABOUT_COMPANY = "asks_about_company"
    ASKS_ABOUT_COMPENSATION = "asks_about_compensation"
    ASKS_ABOUT_LOCATION_OR_WORK_MODE = "asks_about_location_or_work_mode"
    ASKS_ABOUT_VISA_OR_ELIGIBILITY = "asks_about_visa_or_eligibility"
    ASKS_IF_AI_OR_AUTOMATED = "asks_if_ai_or_automated"
    SKEPTICAL_OR_DISTRUSTFUL = "skeptical_or_distrustful"
    RAISES_OBJECTION = "raises_objection"
    BUSY_OR_NOT_READY = "busy_or_not_ready"
    CLEAR_REJECTION = "clear_rejection"
    DO_NOT_CONTACT = "do_not_contact"
    HOSTILE = "hostile"
    UNCLEAR = "unclear"


class Sentiment(StrEnum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    MIXED = "mixed"


class EngagementLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ConversationStage(StrEnum):
    INITIAL_OUTREACH = "initial_outreach"
    DISCOVERY = "discovery"
    INFORMATION_EXCHANGE = "information_exchange"
    OBJECTION_HANDLING = "objection_handling"
    NEXT_STEP = "next_step"
    PAUSED = "paused"
    CLOSED = "closed"


class AgentAction(StrEnum):
    INTRODUCE_OPPORTUNITY = "introduce_opportunity"
    EXPLAIN_CANDIDATE_RELEVANCE = "explain_candidate_relevance"
    ANSWER_CANDIDATE_QUESTION = "answer_candidate_question"
    ASK_DISCOVERY_QUESTION = "ask_discovery_question"
    ADDRESS_OBJECTION = "address_objection"
    CLARIFY_MISSING_INFORMATION = "clarify_missing_information"
    SUGGEST_NEXT_STEP = "suggest_next_step"
    PAUSE_RESPECTFULLY = "pause_respectfully"
    DISCLOSE_AI_IDENTITY = "disclose_ai_identity"
    GRACEFULLY_EXIT = "gracefully_exit"


class OutreachStage(StrEnum):
    INITIAL_OUTREACH = "initial_outreach"
    FOLLOW_UP = "follow_up"
    FINAL_CLOSEOUT = "final_closeout"


class MessageRole(StrEnum):
    AGENT = "agent"
    CANDIDATE = "candidate"
