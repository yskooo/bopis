window.BOPIS_TASK_RULES = {
  "generated_utc": "2026-09-24T20:33:57.901671+00:00",
  "confidence_margin": 0.15,
  "fallback_task": "general_qa",
  "gate_penalty": 0.35,
  "closed_qa_gate_bonus": 2.0,
  "context_question_pattern": "\\?|^\\s*(who|what|when|where|why|how|which)\\b",
  "context_categories": [
    "closed_qa",
    "summarization",
    "information_extraction"
  ],
  "no_context_categories": [
    "open_qa",
    "classification",
    "creative_writing",
    "brainstorming",
    "general_qa"
  ],
  "task_keys": [
    "open_qa",
    "closed_qa",
    "summarization",
    "classification",
    "creative_writing",
    "brainstorming",
    "information_extraction",
    "general_qa"
  ],
  "tasks": {
    "open_qa": {
      "label": "Open QA",
      "sensitivity": "Low",
      "has_context": false
    },
    "closed_qa": {
      "label": "Closed QA",
      "sensitivity": "High",
      "has_context": true
    },
    "summarization": {
      "label": "Summarization",
      "sensitivity": "Low",
      "has_context": true
    },
    "classification": {
      "label": "Classification",
      "sensitivity": "Low",
      "has_context": false
    },
    "creative_writing": {
      "label": "Creative Writing",
      "sensitivity": "Low",
      "has_context": false
    },
    "brainstorming": {
      "label": "Brainstorming",
      "sensitivity": "Low",
      "has_context": false
    },
    "information_extraction": {
      "label": "Information Extraction",
      "sensitivity": "High",
      "has_context": true
    },
    "general_qa": {
      "label": "General QA",
      "sensitivity": "Medium",
      "has_context": false
    }
  },
  "priors": {
    "open_qa": {
      "F16": 0.6,
      "Q8_0": 0.2,
      "Q4_K_M": 0.2
    },
    "closed_qa": {
      "F16": 0.8,
      "Q8_0": 0.15,
      "Q4_K_M": 0.05
    },
    "summarization": {
      "F16": 0.55,
      "Q8_0": 0.225,
      "Q4_K_M": 0.225
    },
    "classification": {
      "F16": 0.45,
      "Q8_0": 0.275,
      "Q4_K_M": 0.275
    },
    "creative_writing": {
      "F16": 0.6,
      "Q8_0": 0.2,
      "Q4_K_M": 0.2
    },
    "brainstorming": {
      "F16": 0.45,
      "Q8_0": 0.275,
      "Q4_K_M": 0.275
    },
    "information_extraction": {
      "F16": 0.8,
      "Q8_0": 0.15,
      "Q4_K_M": 0.05
    },
    "general_qa": {
      "F16": 0.65,
      "Q8_0": 0.21,
      "Q4_K_M": 0.14
    }
  },
  "rules": [
    {
      "category": "summarization",
      "weight": 3.0,
      "pattern": "^\\s*(please\\s+)?(summarize|summarise|sum up)\\b",
      "label": "lead:summarize"
    },
    {
      "category": "summarization",
      "weight": 2.0,
      "pattern": "\\b(summary|summarize|summarise)\\b",
      "label": "kw:summary"
    },
    {
      "category": "summarization",
      "weight": 2.0,
      "pattern": "\\b(tl;?dr|in a nutshell|condense)\\b",
      "label": "kw:tldr"
    },
    {
      "category": "summarization",
      "weight": 2.5,
      "pattern": "\\b(key|main|important)\\s+(points?|findings?|takeaways?|ideas?)\\b",
      "label": "kw:key-points"
    },
    {
      "category": "summarization",
      "weight": 2.0,
      "pattern": "\\bin (your|one|a few|fewer) own words\\b",
      "label": "kw:own-words"
    },
    {
      "category": "summarization",
      "weight": 2.0,
      "pattern": "\\b(shorten|abbreviate|paraphrase|rephrase)\\b",
      "label": "kw:shorten"
    },
    {
      "category": "summarization",
      "weight": 2.0,
      "pattern": "\\bwhat (is|was) (this|the) (passage|text|article|paragraph) about\\b",
      "label": "kw:what-about"
    },
    {
      "category": "summarization",
      "weight": 1.5,
      "pattern": "\\bgive me a (brief|short|quick|one)\\b",
      "label": "kw:give-brief"
    },
    {
      "category": "classification",
      "weight": 3.0,
      "pattern": "^\\s*(please\\s+)?(classify|categorize|categorise|label)\\b",
      "label": "lead:classify"
    },
    {
      "category": "classification",
      "weight": 2.0,
      "pattern": "\\b(classify|categorize|categorise)\\b",
      "label": "kw:classify"
    },
    {
      "category": "classification",
      "weight": 2.0,
      "pattern": "\\bwhich (category|class|type|group)\\b",
      "label": "kw:which-category"
    },
    {
      "category": "classification",
      "weight": 2.0,
      "pattern": "\\b(true or false|positive or negative|yes or no)\\b",
      "label": "kw:binary"
    },
    {
      "category": "classification",
      "weight": 1.5,
      "pattern": "\\bis (this|it|the following)\\s+\\w+\\s+or\\s+\\w+",
      "label": "kw:a-or-b"
    },
    {
      "category": "classification",
      "weight": 1.5,
      "pattern": "\\b(identify|determine|tell me) whether\\b",
      "label": "kw:whether"
    },
    {
      "category": "classification",
      "weight": 1.0,
      "pattern": "\\b(sort|group|bucket) (the|these|those|following)\\b",
      "label": "kw:sort"
    },
    {
      "category": "information_extraction",
      "weight": 3.0,
      "pattern": "^\\s*(please\\s+)?extract\\b",
      "label": "lead:extract"
    },
    {
      "category": "information_extraction",
      "weight": 2.0,
      "pattern": "\\bextract\\b",
      "label": "kw:extract"
    },
    {
      "category": "information_extraction",
      "weight": 2.0,
      "pattern": "\\b(from|using) the (passage|text|paragraph|article|reference)\\b",
      "label": "kw:from-passage"
    },
    {
      "category": "information_extraction",
      "weight": 1.5,
      "pattern": "\\b(list|give me) (all|the|every)\\b.*\\b(mentioned|listed|named|referenced)\\b",
      "label": "kw:list-mentioned"
    },
    {
      "category": "information_extraction",
      "weight": 1.5,
      "pattern": "\\bpull (out|from)\\b",
      "label": "kw:pull-out"
    },
    {
      "category": "information_extraction",
      "weight": 1.0,
      "pattern": "\\bwhat are the (names?|dates?|numbers?|values?)\\b",
      "label": "kw:what-are-names"
    },
    {
      "category": "brainstorming",
      "weight": 3.0,
      "pattern": "^\\s*(please\\s+)?brainstorm\\b",
      "label": "lead:brainstorm"
    },
    {
      "category": "brainstorming",
      "weight": 2.5,
      "pattern": "\\bbrainstorm\\b",
      "label": "kw:brainstorm"
    },
    {
      "category": "brainstorming",
      "weight": 3.0,
      "pattern": "\\b(give|list|name|suggest|tell)\\s+(me\\s+)?(some|a few|several|\\d+)\\b",
      "label": "kw:give-some"
    },
    {
      "category": "brainstorming",
      "weight": 4.5,
      "pattern": "\\bwhat are some\\b",
      "label": "kw:what-are-some"
    },
    {
      "category": "brainstorming",
      "weight": 2.0,
      "pattern": "\\b(ideas?|suggestions?|options?)\\s+(for|to|about|on)\\b",
      "label": "kw:ideas-for"
    },
    {
      "category": "brainstorming",
      "weight": 2.0,
      "pattern": "\\bways to\\b",
      "label": "kw:ways-to"
    },
    {
      "category": "brainstorming",
      "weight": 2.0,
      "pattern": "^\\s*(please\\s+)?(list|name|suggest|recommend)\\b",
      "label": "lead:list"
    },
    {
      "category": "brainstorming",
      "weight": 1.5,
      "pattern": "\\b(recommend|recommendations?)\\b",
      "label": "kw:recommend"
    },
    {
      "category": "brainstorming",
      "weight": 1.5,
      "pattern": "\\bwhat should i\\b",
      "label": "kw:what-should-i"
    },
    {
      "category": "brainstorming",
      "weight": 1.0,
      "pattern": "\\ba list of\\b",
      "label": "kw:a-list-of"
    },
    {
      "category": "creative_writing",
      "weight": 3.0,
      "pattern": "^\\s*(please\\s+)?(write|compose|draft|craft)\\s+(a|an|me)\\b",
      "label": "lead:write-a"
    },
    {
      "category": "creative_writing",
      "weight": 2.5,
      "pattern": "\\b(poem|story|haiku|sonnet|limerick|screenplay|lyrics?)\\b",
      "label": "kw:literary-form"
    },
    {
      "category": "creative_writing",
      "weight": 2.0,
      "pattern": "\\bwrite (a|an)\\s+(short\\s+)?(story|essay|poem|letter|song|dialogue)\\b",
      "label": "kw:write-form"
    },
    {
      "category": "creative_writing",
      "weight": 1.5,
      "pattern": "\\b(imagine|pretend|fictional|make up)\\b",
      "label": "kw:imagine"
    },
    {
      "category": "creative_writing",
      "weight": 1.0,
      "pattern": "\\bin the style of\\b",
      "label": "kw:in-style-of"
    },
    {
      "category": "open_qa",
      "weight": 2.0,
      "pattern": "^\\s*(who|what|when|where|why|how|which)\\b",
      "label": "lead:interrogative"
    },
    {
      "category": "open_qa",
      "weight": 1.5,
      "pattern": "^\\s*(what|who) (is|are|was|were)\\b",
      "label": "kw:what-is"
    },
    {
      "category": "open_qa",
      "weight": 1.0,
      "pattern": "\\b(explain|describe|tell me about)\\b",
      "label": "kw:explain"
    },
    {
      "category": "open_qa",
      "weight": 1.0,
      "pattern": "\\bhow (do|does|did|can)\\b",
      "label": "kw:how-do"
    },
    {
      "category": "closed_qa",
      "weight": 1.5,
      "pattern": "\\b(according to|based on|per) the (passage|text|paragraph|article|reference|above)\\b",
      "label": "kw:according-to"
    },
    {
      "category": "closed_qa",
      "weight": 1.0,
      "pattern": "\\bgiven (the|this) (passage|text|paragraph|article|context|reference)\\b",
      "label": "kw:given-passage"
    }
  ],
  "measured_accuracy": {
    "n": 15011.0,
    "eight_way": 0.497,
    "qa_merged": 0.643,
    "sensitivity_tier": 0.675
  },
  "note": "Rule-based classifier. 49.7% exact 8-way accuracy on Dolly 15k; 67.5% on the quality-sensitivity tier that drives the prior. The prior weights only the 10 BO seed draws, not the selection of x*."
};
