"""
LangChain ReAct Agent with Firecrawl Tools
Uses LLM to decide what to search/scrape for market sizing

Cost: ~$0.20-0.30 per segment (vs $2.91 with Firecrawl Agent API)
"""

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from firecrawl import FirecrawlApp
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
import json
from datetime import datetime
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Global counter to track tool usage and detect loops
_tool_call_counter = 0
_recent_queries = []  # Track recent queries to detect loops
_max_identical_queries = 3  # Warn after 3 identical queries

# ============================================================================
# CONFIGURATION
# ============================================================================

FIRECRAWL_API_KEY = "fc-3b076c020ac74c818afe09842b460bba"
GEMINI_API_KEY = "AIzaSyC-_ef8P9bFtbec3RNRyYisyNQ2eALV4EE"

firecrawl = FirecrawlApp(api_key=FIRECRAWL_API_KEY)

# ============================================================================
# PYDANTIC SCHEMAS (Same as before)
# ============================================================================

class Source(BaseModel):
    """Source URL with description"""
    url: str
    title: str
    relevance: str = Field(description="Why this source is relevant")

class PopulationData(BaseModel):
    """Struggle-aware population calculation"""
    total_addressable: int
    struggle_aware_count: int
    frequency_filter: Literal["daily", "weekly", "monthly", "quarterly"]
    intensity_filter: Literal["critical", "high", "medium", "low"]
    calculation_method: str
    sources: List[Source] = Field(min_length=1)

class PricingComparable(BaseModel):
    """Competitor pricing information"""
    solution_name: str
    annual_price: float
    source_url: str
    positioning: str

class PricingTier(BaseModel):
    """Pricing tier recommendation"""
    tier_name: Literal["Low", "Mid", "High"]
    annual_price: float
    rationale: str
    comparables: List[PricingComparable] = Field(min_length=2)
    sam_revenue: float
    capture_rate: str
    som_year_1: float

class MarketSizingResult(BaseModel):
    """Complete market sizing analysis"""
    segment_name: str
    location: str
    population_data: PopulationData
    pricing_tiers: List[PricingTier] = Field(min_length=3, max_length=3)
    recommended_tier: Literal["Low", "Mid", "High"]
    recommendation_reason: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class UserSegment(BaseModel):
    """Individual user segment with full analysis"""
    segment_name: str
    description: str
    context_circumstance: str
    key_constraints: List[str] = Field(min_length=2, max_length=5)
    buyer_type: Literal["individual-self-serve", "manager-approved", "procurement-committee"]
    struggle_frequency: Literal["daily", "weekly", "monthly", "quarterly", "occasional"]
    struggle_intensity: Literal["critical", "high", "medium", "low"]
    existing_alternatives: str
    priority_level: Literal["primary", "secondary", "alternative"]
    priority_rationale: str = Field(description="3-5 sentences explaining priority assignment with data")
    product_vibe: str = Field(description="Emotional/functional personality of product for this segment")
    possible_features: List[str] = Field(min_length=5, max_length=8)
    behavioral_evidence: str = Field(description="Evidence of the behavior with source URLs")
    pain_intensity_evidence: str = Field(description="Evidence of pain with source URLs")
    market_data_sources: List[Source] = Field(min_length=2)

class SegmentGenerationResult(BaseModel):
    """Complete segment generation with 8-12 segments"""
    segments: List[UserSegment] = Field(min_length=8, max_length=12)
    total_segments: int
    primary_count: int = Field(default=1)
    secondary_count: int = Field(default=2)
    alternative_count: int

# ============================================================================
# FIRECRAWL TOOLS (for LangChain Agent)
# ============================================================================

@tool
def firecrawl_search(query: str, limit: int = 10) -> str:
    """
    Search the web using Firecrawl. Returns URLs and snippets.
    Use this to find relevant sources for market data, competitor info, etc.
    
    Args:
        query: Search query (e.g., "freelancer market size United States 2025")
        limit: Number of results (default 10, max 100)
    
    Returns:
        JSON string with search results containing URLs, titles, descriptions
        
    Cost: 2 credits per 10 results
    """
    global _tool_call_counter, _recent_queries
    
    # Increment counter for tracking
    _tool_call_counter += 1
    
    # Check for duplicate queries (infinite loop detection)
    _recent_queries.append(query)
    if len(_recent_queries) > 10:
        _recent_queries.pop(0)
    
    identical_count = _recent_queries.count(query)
    if identical_count >= _max_identical_queries:
        print(f"\n⚠️  [LOOP WARNING] Query repeated {identical_count} times - returning empty results")
        print(f"   Suggestion: Try different keywords or make estimates")
        # Return empty results instead of error - let agent continue
        return json.dumps({
            'success': True,
            'query': query,
            'results_count': 0,
            'results': [],
            'credits_used': 0,
            'note': f'Query repeated {identical_count} times. Try different approach.'
        })
    
    start_time = time.time()
    print(f"\n🔍 [TOOL #{_tool_call_counter}] firecrawl_search STARTED at {datetime.now().strftime('%H:%M:%S')}")
    print(f"   Query: {query[:80]}{'...' if len(query) > 80 else ''}")
    print(f"   Limit: {limit}")
    if identical_count > 1:
        print(f"   ⚠️  WARNING: Query seen {identical_count} times before!")
    
    try:
        api_start = time.time()
        result = firecrawl.search(
            query=query,
            limit=min(limit, 20),  # Cap at 20 to control costs
        )
        api_duration = time.time() - api_start
        print(f"   ⏱️  Firecrawl API responded in {api_duration:.2f}s")
        
        # Handle SearchData object (not a dict - use attributes)
        success = getattr(result, 'success', True)
        if not success:
            error_msg = getattr(result, 'error', 'Unknown error')
            print(f"   ⚠️  Search returned error: {error_msg}")
            print(f"   📊 Credits: 0 (failed call - continuing anyway)")
            # Return empty results with 0 credits - let agent continue
            return json.dumps({
                'success': True,
                'query': query,
                'results_count': 0,
                'results': [],
                'credits_used': 0,
                'note': f'Search failed: {error_msg}. Try different query.'
            })
        
        # Extract relevant data from SearchData object
        data = getattr(result, 'data', None)
        if data:
            web_results = getattr(data, 'web', [])
            
            if not web_results or len(web_results) == 0:
                credits_used = getattr(result, 'creditsUsed', 0)
                print(f"   ⚠️  No results found for this query")
                print(f"   📊 Credits: {credits_used} (empty result - continuing)")
                # Return empty results - let agent continue
                return json.dumps({
                    'success': True,
                    'query': query,
                    'results_count': 0,
                    'results': [],
                    'credits_used': credits_used,
                    'note': 'No results found. Try different keywords.'
                })
            
            simplified = []
            for item in web_results[:limit]:
                # Handle each web result (also might be an object)
                url = getattr(item, 'url', '') if hasattr(item, 'url') else item.get('url', '')
                title = getattr(item, 'title', '') if hasattr(item, 'title') else item.get('title', '')
                description = getattr(item, 'description', '') if hasattr(item, 'description') else item.get('description', '')
                
                simplified.append({
                    'url': url,
                    'title': title,
                    'description': description
                })
            
            total_duration = time.time() - start_time
            credits = getattr(result, 'creditsUsed', 2)
            print(f"   ✅ Found {len(simplified)} results")
            print(f"   💰 Credits used: {credits}")
            print(f"   ⏱️  Total tool time: {total_duration:.2f}s")
            print(f"   🏁 [TOOL] firecrawl_search COMPLETED at {datetime.now().strftime('%H:%M:%S')}\n")
            
            return json.dumps({
                'success': True,
                'query': query,
                'results_count': len(simplified),
                'results': simplified,
                'credits_used': credits,
                'execution_time_seconds': round(total_duration, 2)
            }, indent=2)
        else:
            duration = time.time() - start_time
            print(f"   ⚠️  Search returned no data")
            print(f"   ⏱️  Tool time: {duration:.2f}s")
            print(f"   📊 Credits: 0 (no data - continuing anyway)\n")
            # Return empty results - let agent continue
            return json.dumps({
                'success': True,
                'query': query,
                'results_count': 0,
                'results': [],
                'credits_used': 0,
                'note': 'No data returned. Try different approach.'
            })
    except Exception as e:
        duration = time.time() - start_time
        print(f"   ⚠️  Exception occurred: {str(e)[:100]}")
        print(f"   ⏱️  Tool time: {duration:.2f}s")
        print(f"   📊 Credits: 0 (error - continuing anyway)\n")
        # Return empty results - let agent continue
        return json.dumps({
            'success': True,
            'query': query,
            'results_count': 0,
            'results': [],
            'credits_used': 0,
            'note': f'Error: {str(e)[:50]}. Try different approach.'
        })


@tool
def firecrawl_scrape(url: str) -> str:
    """
    Scrape a specific URL to get full content using Firecrawl.
    Use this after finding relevant URLs via search.
    
    Args:
        url: The URL to scrape
    
    Returns:
        JSON string with markdown content and metadata
        
    Cost: 1 credit per page
    """
    global _tool_call_counter
    
    # Increment counter for tracking
    _tool_call_counter += 1
    
    start_time = time.time()
    print(f"\n📄 [TOOL #{_tool_call_counter}] firecrawl_scrape STARTED at {datetime.now().strftime('%H:%M:%S')}")
    print(f"   URL (EXACT): {url}")
    if len(url) > 100:
        print(f"   URL (display): {url[:100]}...")
    
    try:
        api_start = time.time()
        result = firecrawl.scrape(
            url=url,
            formats=['markdown']
        )
        api_duration = time.time() - api_start
        print(f"   ⏱️  Firecrawl API responded in {api_duration:.2f}s")
        
        # Handle ScrapeData object (not a dict - use attributes)
        success = getattr(result, 'success', True)
        if success:
            # Get data from result object
            data = getattr(result, 'data', {})
            markdown = getattr(data, 'markdown', '') if hasattr(data, 'markdown') else data.get('markdown', '') if isinstance(data, dict) else ''
            
            # Truncate to first 5000 chars to avoid overwhelming the LLM
            truncated = markdown[:5000]
            if len(markdown) > 5000:
                truncated += "\n\n[Content truncated for length...]"
            
            total_duration = time.time() - start_time
            content_length = len(markdown)
            print(f"   ✅ Scraped {content_length:,} characters")
            print(f"   📝 Returned {len(truncated):,} characters (truncated)")
            print(f"   💰 Credits used: 1")
            print(f"   ⏱️  Total tool time: {total_duration:.2f}s")
            print(f"   🏁 [TOOL] firecrawl_scrape COMPLETED at {datetime.now().strftime('%H:%M:%S')}\n")
            
            return json.dumps({
                'success': True,
                'url': url,
                'content': truncated,
                'full_length': content_length,
                'credits_used': 1,
                'execution_time_seconds': round(total_duration, 2)
            }, indent=2)
        else:
            duration = time.time() - start_time
            error = getattr(result, 'error', 'Scrape failed')
            print(f"   ⚠️  Scrape failed: {error[:80]}")
            print(f"   ⏱️  Tool time: {duration:.2f}s")
            print(f"   📊 Credits: 0 (failed - continuing anyway)\n")
            # Return empty content - let agent continue
            return json.dumps({
                'success': True,
                'url': url,
                'content': '',
                'full_length': 0,
                'credits_used': 0,
                'note': f'Scrape failed: {error[:50]}. Try different URL.',
                'execution_time_seconds': round(duration, 2)
            }, indent=2)
    except Exception as e:
        duration = time.time() - start_time
        print(f"   ⚠️  Exception occurred: {str(e)[:100]}")
        print(f"   ⏱️  Tool time: {duration:.2f}s")
        print(f"   📊 Credits: 0 (error - continuing anyway)\n")
        # Return empty content - let agent continue
        return json.dumps({
            'success': True,
            'url': url,
            'content': '',
            'full_length': 0,
            'credits_used': 0,
            'note': f'Error: {str(e)[:50]}. Try different URL.',
            'execution_time_seconds': round(duration, 2)
        }, indent=2)


# ============================================================================
# LANGCHAIN AGENT SETUP
# ============================================================================

def create_market_sizing_agent():
    """Create a LangChain agent with Firecrawl tools"""
    
    # Initialize Gemini model
    model = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        google_api_key=GEMINI_API_KEY,
        temperature=0.1
    )
    
    # Define tools
    tools = [firecrawl_search, firecrawl_scrape]
    
    # System prompt
    system_prompt = """You are a FAST, EFFICIENT market research agent specialized in TWO-PHASE startup market analysis:

PHASE 1: USER SEGMENT GENERATION (8-12 segments with priorities)
PHASE 2: MARKET SIZING (for top 3 segments only)

⚡ SPEED & EFFICIENCY REQUIREMENTS:
- WORK FAST: Each tool call should serve a clear purpose
- MINIMIZE TOOL CALLS: Get maximum value from each search/scrape
- NO OVER-THINKING: Make quick decisions based on available data
- PARALLEL THINKING: Plan all searches upfront, don't do one at a time
- TARGET TIME: Complete each segment in 20-30 seconds
- IF SEARCH FAILS: Try a DIFFERENT query, don't retry the same one
- IF NO DATA FOUND: Make educated estimates and document assumptions

🚫 CRITICAL URL RULES - READ CAREFULLY:
❌ NEVER modify, edit, or change URLs returned by firecrawl_search
❌ NEVER create your own URLs or make up URLs
❌ NEVER add or remove characters from URLs (no http → https changes, no trailing slashes, nothing)
✅ Use URLs EXACTLY as returned by firecrawl_search tool - copy them character-by-character
✅ If you need to scrape a URL, use the EXACT 'url' field from search results
✅ URLs must be IDENTICAL to what firecrawl_search returned - no modifications allowed

Example CORRECT usage:
  Search returns: {'url': 'https://example.com/data?id=123'}
  You scrape: 'https://example.com/data?id=123'  ← EXACT COPY

Example WRONG usage:
  Search returns: {'url': 'https://example.com/data?id=123'}
  You scrape: 'https://example.com/data' ← WRONG! Query removed
  You scrape: 'http://example.com/data?id=123' ← WRONG! Changed https to http
  You scrape: 'https://example.com/data?id=123/' ← WRONG! Added trailing slash

🚨 COST AWARENESS:
- firecrawl_search: 2 credits per 10 results (~$0.06)
- firecrawl_scrape: 1 credit per page (~$0.03)
- BE STRATEGIC: Search broadly first, then scrape only the most promising 2-3 URLs (NOT 5+)
- DO NOT scrape every URL you find - evaluate search results titles/descriptions first
- AVOID redundant searches - use what you already learned
- STOP when you have enough data - don't chase perfection

YOUR RESEARCH STRATEGY:

PHASE 1 STRATEGY (Segment Generation):
1. Search for: "[idea/JTBD] user segments market research"
2. Search for: "[idea/JTBD] customer types personas"
3. Scrape top 2-3 most relevant URLs
4. Search for: "[idea/JTBD] market size statistics [location]"
5. Scrape 1-2 market reports
6. With this data, generate 8-12 segments with priorities (1 PRIMARY, 2 SECONDARY, rest ALTERNATIVE)

PHASE 2 STRATEGY (Market Sizing for top 3 only) - WORK FAST:
For EACH of the 3 prioritized segments (aim for 20-30 seconds each):
1. Search: "[segment] market size [location] 2024 2025" → Pick BEST URL from results
2. Scrape: ONLY the #1 most promising source (not 2, not 3 - just 1 unless critical data missing)
3. Search: "[segment] pricing comparison competitors [location]" → Evaluate results
4. Scrape: ONLY if pricing not clear from search results (1 page max)
5. Calculate: Struggle-aware population + 3 pricing tiers

⚡ SPEED TIPS:
- Use search result descriptions - they often have key numbers WITHOUT scraping
- If you see market size in search snippet, you may not need to scrape
- Scrape ONLY when search results lack specific data
- Make educated estimates when exact data unavailable (document assumptions)

CRITICAL RULES:
✅ PHASE 1: Generate ALL segments (8-12) with priorities before moving to PHASE 2
✅ PHASE 2: Only research market sizing for PRIMARY + 2 SECONDARY (skip alternatives)
✅ Focus on STRUGGLE-AWARE population (not total market)
✅ Every number needs source URLs
✅ Use [location]-specific data only
✅ Filter by frequency (daily/weekly) and intensity (critical/high)
✅ Be COST-CONSCIOUS: Aim for ~15-25 total tool calls for efficiency
✅ URLs: Use EXACT URLs from firecrawl_search - NEVER modify them

🚨 ERROR HANDLING RULES:
❌ NEVER retry the exact same query if it fails - try different keywords
❌ NEVER modify URLs - use exact URLs from search results
❌ NEVER create fake URLs - only use URLs returned by firecrawl_search
❌ If searches return no results, make educated estimates with available data
❌ If stuck, MOVE ON and complete analysis with what you have
✅ Document any assumptions or estimates you make
✅ Copy URLs character-by-character from search results to scrape calls

TOOL USAGE LIMITS (Stay within budget AND time):
- PHASE 1: ~4-6 searches, ~2-4 scrapes (10-14 credits = $0.30-0.42) | Target: 30-45 seconds
- PHASE 2: ~3-6 searches (1-2 per segment), ~3-6 scrapes (1-2 per segment) (9-18 credits = $0.27-0.54) | Target: 60-90 seconds
- TOTAL TARGET: 19-32 credits = $0.57-0.96 per full analysis (vs $2.91 with Firecrawl Agent)

⚡ TIME TARGET: Complete entire research in 90-135 seconds (1.5-2.25 minutes)
   - Each tool call averages 3-5 seconds
   - Minimize round-trips between thinking and calling tools
   - Batch your thinking: plan multiple searches, then execute

OUTPUT FORMAT:
When you complete research, structure your findings according to the provided JSON schemas."""
    
    # Create agent with verbose logging enabled
    print("\n🤖 [AGENT] Initializing LangChain agent with verbose mode...")
    print("   Model: gemini-3-flash-preview")
    print("   Tools: firecrawl_search, firecrawl_scrape")
    print("   System: Optimized for speed and efficiency\n")
    
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt
    )
    
    return agent


# ============================================================================
# MAIN RESEARCH FUNCTIONS
# ============================================================================

def generate_segments(idea: str, jtbd: str, location: str = "United States"):
    """
    PHASE 1: Generate 8-12 user segments with priorities
    
    Args:
        idea: Product/service idea
        jtbd: Jobs to be done narrative (can be string or dict)
        location: Geographic location for data
        
    Returns:
        SegmentGenerationResult (Pydantic model)
    """
    # Reset global counters for new run
    global _tool_call_counter, _recent_queries
    _tool_call_counter = 0
    _recent_queries = []
    
    agent = create_market_sizing_agent()
    
    # Handle JTBD as string or dict
    if isinstance(jtbd, dict):
        jtbd_narrative = jtbd.get('narrative', str(jtbd))
    else:
        jtbd_narrative = jtbd
    
    # Construct segment generation query
    query = f"""# PHASE 1: USER SEGMENT GENERATION

**Context:**
- Product Idea: {idea}
- JTBD: {jtbd_narrative}
- Target Location: {location}

**Your Task:** Generate 8-12 distinct user segments who experience this specific struggle.

**JTBD Format Explained:**
The JTBD follows: [user_segment] in [situation] need to [JTBD] so that they can [outcome]

This represents the specific struggle a user faces. Find sub-groups who experience this struggle with different:
- Intensities
- Contexts
- Constraints  
- Willingness to pay

## Research Requirements (USE TOOLS CAREFULLY - THEY COST MONEY):

**Step 1: Broad Market Research (2-3 searches, 2-3 scrapes)**
- Search: "[idea] user personas customer segments"
- Search: "[idea] target market analysis {location}"
- Scrape: Top 2-3 most relevant sources
  ⚠️ CRITICAL: Use EXACT URLs from search results - do not modify them
- Look for: Different user types, pain points, market size data

**Step 2: Segment-Specific Data (1-2 searches, 1-2 scrapes)**
- Search: "[idea] market opportunity {location} statistics"
- Scrape: Industry reports, market research
  ⚠️ CRITICAL: Use EXACT URLs from search results - do not modify them
- Look for: Population numbers, growth trends, adoption barriers

🚫 URL RULES:
When calling firecrawl_scrape, copy the 'url' field EXACTLY from search results.
Do NOT modify, change, or edit URLs in any way. Use them character-by-character as returned.

## Segment Variation Dimensions:

1. **Context/Circumstance**: When/where does struggle happen?
2. **Key Constraints**: What blocks them? (time, budget, compliance, skills)
3. **Buyer Type**: Who decides? (individual/manager/committee)
4. **Struggle Frequency**: How often? (daily/weekly/monthly/quarterly)
5. **Struggle Intensity**: How painful? (critical/high/medium/low)
6. **Existing Alternatives**: What do they use today?

## Prioritization (CRITICAL):

After identifying all 8-12 segments, assign priority levels:

1. **PRIMARY (1 segment only)**: Most promising based on:
   - Market size and accessibility
   - Pain intensity and willingness to pay
   - Speed to revenue/traction
   - Strategic value (word-of-mouth, network effects)

2. **SECONDARY (2 segments)**: Next two most promising that complement primary

3. **ALTERNATIVE (remaining)**: Viable segments for future expansion

## For Each Segment Provide:

- segment_name: Clear, specific name
- description: 2-3 sentences
- context_circumstance: When/where struggle occurs
- key_constraints: 2-5 specific blockers
- buyer_type: Who makes purchase decision
- struggle_frequency: How often faced
- struggle_intensity: Pain level
- existing_alternatives: What they use today
- priority_level: primary/secondary/alternative
- priority_rationale: 3-5 sentences with data justifying priority
- product_vibe: Emotional/functional personality
- possible_features: 5-8 features addressing constraints
- behavioral_evidence: Evidence with source URLs
- pain_intensity_evidence: Evidence with source URLs
- market_data_sources: List of source URLs with descriptions

**CRITICAL**: Every data point needs source URLs. Use real market research, surveys, reports.

**Output Format:** Provide data as JSON matching SegmentGenerationResult schema with 8-12 segments."""
    
    print("\n" + "="*80)
    print("🔍 PHASE 1: USER SEGMENT GENERATION")
    print("="*80)
    print(f"Idea: {idea}")
    print(f"Location: {location}")
    print("\n🤖 LangChain agent starting research...")
    print("   Target: 8-12 segments (1 PRIMARY, 2 SECONDARY, rest ALTERNATIVE)")
    print("   Budget: ~10-14 credits ($0.30-0.42)")
    print("   ⏱️  Time Target: 30-45 seconds")
    print("\n" + "="*80)
    print("📝 AGENT ACTIVITY LOG (Verbose Mode Enabled)")
    print("="*80)
    print("Watch below for real-time tool calls and agent reasoning...")
    print("⚡ SPEED MODE: Agent optimized to minimize tool calls and work efficiently\n")
    
    phase1_start = time.time()
    print(f"⏱️  [PHASE 1] Started at {datetime.now().strftime('%H:%M:%S')}\n")
    
    result = agent.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    
    # Print verbose agent activity and track credits + tokens + tool usage
    total_credits = 0
    total_input_tokens = 0
    total_output_tokens = 0
    search_count = 0
    scrape_count = 0
    search_results_log = []  # Track all search results
    
    print("\n🔍 Agent Messages:")
    for i, msg in enumerate(result["messages"], 1):
        if hasattr(msg, 'type'):
            if msg.type == 'ai':
                # Extract token usage from AI messages
                if hasattr(msg, 'usage_metadata'):
                    usage = msg.usage_metadata
                    total_input_tokens += usage.get('input_tokens', 0)
                    total_output_tokens += usage.get('output_tokens', 0)
                
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_name = tc['name']
                        # Count tool usage
                        if tool_name == 'firecrawl_search':
                            search_count += 1
                        elif tool_name == 'firecrawl_scrape':
                            scrape_count += 1
                        
                        elapsed = time.time() - phase1_start
                        print(f"\n   [{i}] 🛠️  Agent calling: {tool_name} (Tool #{_tool_call_counter + 1}) (at {elapsed:.1f}s)")
                        if 'query' in tc['args']:
                            query_text = tc['args']['query']
                            print(f"       Query: {query_text[:80]}...")
                            # Check if this is a duplicate
                            if query_text in _recent_queries:
                                print(f"       ⚠️  WARNING: This query was used before!")
                        elif 'url' in tc['args']:
                            print(f"       URL: {tc['args']['url'][:80]}...")
                elif hasattr(msg, 'content') and msg.content:
                    print(f"   [{i}] 💭 Agent reasoning: {str(msg.content)[:150]}...")
            elif msg.type == 'tool':
                # Extract credits from tool result
                try:
                    tool_result = json.loads(msg.content)
                    credits = tool_result.get('credits_used', 0)
                    exec_time = tool_result.get('execution_time_seconds', 0)
                    total_credits += credits
                    elapsed = time.time() - phase1_start
                    success = tool_result.get('success', True)
                    
                    # Track search results for output file
                    if tool_result.get('query'):  # This is a search result
                        search_results_log.append({
                            'query': tool_result.get('query'),
                            'results_count': tool_result.get('results_count', 0),
                            'results': tool_result.get('results', []),
                            'credits_used': credits,
                            'execution_time_seconds': exec_time
                        })
                    
                    if credits == 0:
                        print(f"   [{i}] ⚠️  Tool returned 0 credits (likely failed/empty) | Pipeline at {elapsed:.1f}s")
                        note = tool_result.get('note', '')
                        if note:
                            print(f"       Note: {note[:80]}")
                    elif not success:
                        error = tool_result.get('error', 'Unknown error')
                        print(f"   [{i}] ❌ Tool FAILED: {error[:100]}")
                    else:
                        print(f"   [{i}] ✅ Tool completed in {exec_time:.2f}s (Credits: {credits}) | Pipeline at {elapsed:.1f}s")
                except:
                    elapsed = time.time() - phase1_start
                    print(f"   [{i}] ✅ Tool result received | Pipeline at {elapsed:.1f}s")
    
    final_message = result["messages"][-1].content
    
    # Parse and display structured segments
    print("\n" + "="*80)
    print("📊 STRUCTURED RESULTS - PHASE 1")
    print("="*80)
    try:
        # Extract JSON from markdown code blocks if present
        content_text = final_message
        if isinstance(content_text, list) and len(content_text) > 0:
            content_text = content_text[0].get('text', '')
        
        if '```json' in content_text:
            json_start = content_text.find('```json') + 7
            json_end = content_text.find('```', json_start)
            content_text = content_text[json_start:json_end].strip()
        
        segments_data = json.loads(content_text)
        segments = segments_data.get('segments', [])
        
        print(f"\n✨ Generated {len(segments)} User Segments:\n")
        
        # Group by priority
        primary = [s for s in segments if s.get('priority_level') == 'primary']
        secondary = [s for s in segments if s.get('priority_level') == 'secondary']
        alternative = [s for s in segments if s.get('priority_level') == 'alternative']
        
        if primary:
            print("🎯 PRIMARY (1):")
            for s in primary:
                print(f"   • {s['segment_name']}")
                print(f"     {s['description'][:100]}...")
                print(f"     Frequency: {s['struggle_frequency']} | Intensity: {s['struggle_intensity']}")
                # Show key sources
                if 'market_data_sources' in s and s['market_data_sources']:
                    print(f"     📚 Sources: {len(s['market_data_sources'])} references")
                    for idx, src in enumerate(s['market_data_sources'][:2], 1):
                        print(f"        [{idx}] {src[:60]}...")
        
        if secondary:
            print(f"\n🔸 SECONDARY ({len(secondary)}):")
            for s in secondary:
                print(f"   • {s['segment_name']}")
                print(f"     {s['description'][:100]}...")
                if 'market_data_sources' in s and s['market_data_sources']:
                    print(f"     📚 {len(s['market_data_sources'])} sources")
        
        if alternative:
            print(f"\n⚪ ALTERNATIVE ({len(alternative)}):")
            for s in alternative:
                print(f"   • {s['segment_name']}")
                if 'market_data_sources' in s and s['market_data_sources']:
                    print(f"     📚 {len(s['market_data_sources'])} sources")
    except Exception as e:
        print(f"⚠️  Could not parse structured output: {str(e)[:100]}")
        print(f"Raw output length: {len(str(final_message))} characters")
    
    phase1_duration = time.time() - phase1_start
    
    print("\n" + "="*80)
    print("✅ PHASE 1 COMPLETED!")
    print(f"   ⏱️  Time: {phase1_duration:.1f}s ({phase1_duration/60:.1f} min)")
    print(f"   🔍 Tool Usage: {search_count} searches + {scrape_count} scrapes = {search_count + scrape_count} total")
    print(f"   💰 Credits used: {total_credits} (${total_credits * 0.03:.2f})")
    print(f"   🤖 Tokens: {total_input_tokens:,} input + {total_output_tokens:,} output = {total_input_tokens + total_output_tokens:,} total")
    print("="*80 + "\n")
    
    return {
        "phase": "segment_generation",
        "structured_segments": segments if 'segments' in locals() else None,
        "raw_result": final_message,
        "messages_count": len(result["messages"]),
        "credits_used": total_credits,
        "search_count": search_count,
        "scrape_count": scrape_count,
        "search_results": search_results_log,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "duration_seconds": round(phase1_duration, 2),
        "timestamp": datetime.now().isoformat()
    }


def size_market_for_segments(idea: str, jtbd: str, segments: List[dict], 
                             location: str = "United States"):
    """
    PHASE 2: Market sizing for top 3 segments (PRIMARY + 2 SECONDARY)
    
    Args:
        idea: Product/service idea
        jtbd: Jobs to be done narrative
        segments: List of segment dicts (must include priority_level)
        location: Geographic location
        
    Returns:
        List of market sizing results for top 3 segments
    """
    
    # Filter to top 3 segments
    top_segments = [s for s in segments if s.get('priority_level') in ['primary', 'secondary']][:3]
    
    if len(top_segments) < 3:
        print("⚠️  Warning: Less than 3 top-priority segments found. Sizing available segments...")
    
    print("\n" + "="*80)
    print("💰 PHASE 2: MARKET SIZING (Top 3 Segments Only)")
    print("="*80)
    print(f"Sizing {len(top_segments)} segments:")
    for i, seg in enumerate(top_segments, 1):
        priority = seg.get('priority_level', 'unknown').upper()
        print(f"   {i}. [{priority}] {seg.get('segment_name', 'Unknown')}")
    print(f"\n   Budget per segment: ~3-6 credits ($0.09-0.18)")
    print(f"   Total budget: ~9-18 credits ($0.27-0.54)")
    print(f"   ⏱️  Time per segment: 20-30 seconds target")
    print(f"   ⏱️  Total time target: 60-90 seconds (parallel)")
    print(f"\n   🚀 Running in PARALLEL for faster completion!\n")
    
    phase2_start = time.time()
    agent = create_market_sizing_agent()
    results = []
    
    # Handle JTBD as string or dict
    if isinstance(jtbd, dict):
        jtbd_narrative = jtbd.get('narrative', str(jtbd))
    else:
        jtbd_narrative = jtbd
    
    # Define function to process a single segment
    def process_segment(segment_tuple):
        i, segment = segment_tuple
        segment_name = segment.get('segment_name', 'Unknown')
        segment_desc = segment.get('description', '')
        priority = segment.get('priority_level', 'unknown').upper()
        
        segment_start = time.time()
        
        print(f"\n{'='*80}")
        print(f"📊 Sizing {i}/{len(top_segments)}: [{priority}] {segment_name}")
        print(f"   ⏱️  Target: 20-30 seconds | Max 4-6 tool calls")
        print(f"{'='*80}\n")
        
        query = f"""# PHASE 2: MARKET SIZING FOR SEGMENT (⚡ WORK FAST - 20-30 seconds target)

**Context:**
- Product: {idea}
- JTBD: {jtbd_narrative}
- Segment: {segment_name}
- Description: {segment_desc}
- Location: {location}

**Your Task:** Calculate market size based on STRUGGLE-AWARE population (not total market).

⚡ SPEED REQUIREMENTS:
- Complete in 20-30 seconds
- Max 2-3 tool calls total (not 4-6!)
- Use search results WITHOUT scraping when possible
- Make educated estimates if data incomplete

## Research Strategy (FAST & EFFICIENT):

**Step 1: Population Sizing (1 search + maybe 1 scrape)**
- Search: "{segment_name} market size {location} 2024 2025 statistics"
- Check search snippets FIRST - do they have numbers? If YES, skip scraping!
- Scrape: ONLY if no numbers in search results (pick #1 most promising URL only)
  ⚠️ IMPORTANT: Use the EXACT 'url' from search results - do not modify it
- Look for: Total addressable, prevalence rates

**Step 2: Pricing Research (1 search + maybe 1 scrape)**
- Search: "{segment_name} pricing competitors solutions {location}"
- Check search results - do titles/descriptions mention prices? If YES, use those!
- Scrape: ONLY if prices not visible in search results (1 page max)
  ⚠️ IMPORTANT: Use the EXACT 'url' from search results - do not modify it
- Look for: Current prices, market benchmarks

🚫 URL REMINDER:
- Copy URLs exactly from firecrawl_search results
- Do not change http/https, add/remove slashes, or modify query parameters
- Use the complete URL string as-is

## STEP 1: STRUGGLE-AWARE POPULATION SIZING

**CRITICAL**: Calculate "struggle-aware" population, NOT total market.

1. **Find Total Addressable Population**
   - Total number of [{segment_name}] in {location}
   - Example: "15M freelancers in US (Upwork 2024)"
   - MUST cite source URL

2. **Filter to Struggle-Aware**
   - Apply filters for those who ACTUALLY experience this struggle:
     * Frequency filter (daily/weekly/monthly)
     * Intensity filter (critical/high pain)
     * Geographic filter (if needed)
   - Example: "15M × 60% (weekly struggle) × 45% (high intensity) = 4M"
   - Show calculation logic
   - Cite sources for each percentage

3. **Provide Evidence**
   - Prevalence rate sources
   - Frequency/intensity data sources

## STEP 2: DEFENDABLE PRICING (Real Market Comparables)

1. **Find Market Benchmarks**
   - Find 3-5 comparable solutions {segment_name} in {location} currently pays for
   - Get actual pricing with source URLs
   - Example: "Tool X: $50/mo (source: pricing page URL)"

2. **Calculate Value/ROI**
   - Dollar value of solving this problem in {location}
   - Consider: Local salaries, cost of living, revenue potential
   - Example: "Saves 5 hours/week × $50/hr = $13k annually"

3. **Generate 3 Pricing Scenarios**
   
   **Low Tier (Mass Market):**
   - Annual price (local currency + USD)
   - Rationale: 3-4 sentences comparing to competitors, ROI, why reasonable for {location}
   - Comparable solutions: 2-3 real competitors with prices and URLs
   - SAM revenue: struggle_aware × price
   - SOM Year 1: Realistic capture rate with justification

   **Mid Tier (Value-Based):**
   - [Same structure as Low]

   **High Tier (Premium):**
   - [Same structure as Low]

## CRITICAL REQUIREMENTS:

✅ Every number must have source URL
✅ Use {location}-specific data
✅ Calculate struggle-aware (not total market)
✅ Provide 3+ comparable prices with sources
✅ Show ROI calculation
✅ Be honest about data confidence

❌ Do NOT use total market
❌ Do NOT make up numbers
❌ Do NOT ignore location context

**Output Format:** Provide as JSON matching MarketSizingResult schema."""

        print(f"\n{'='*80}")
        print(f"📝 AGENT ACTIVITY LOG - {segment_name}")
        print(f"{'='*80}")
        print("Watch below for real-time tool calls and agent reasoning...\n")
        
        result = agent.invoke({
            "messages": [{"role": "user", "content": query}]
        })
        
        # Print verbose agent activity and track credits + tokens + tool usage
        segment_credits = 0
        segment_input_tokens = 0
        segment_output_tokens = 0
        segment_search_count = 0
        segment_scrape_count = 0
        search_results_log = []  # Track search results for this segment
        
        print("\n🔍 Agent Messages & Activity:")
        for idx, msg in enumerate(result["messages"], 1):
            if hasattr(msg, 'type'):
                if msg.type == 'ai':
                    # Extract token usage from AI messages
                    if hasattr(msg, 'usage_metadata'):
                        usage = msg.usage_metadata
                        segment_input_tokens += usage.get('input_tokens', 0)
                        segment_output_tokens += usage.get('output_tokens', 0)
                    
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        for tc in msg.tool_calls:
                            tool_name = tc['name']
                            # Count tool usage
                            if tool_name == 'firecrawl_search':
                                segment_search_count += 1
                            elif tool_name == 'firecrawl_scrape':
                                segment_scrape_count += 1
                            
                            elapsed = time.time() - segment_start
                            print(f"\n   [{idx}] 🛠️  Agent calling: {tool_name} (at {elapsed:.1f}s)")
                            if 'query' in tc['args']:
                                print(f"       Query: {tc['args']['query'][:80]}...")
                            elif 'url' in tc['args']:
                                print(f"       URL: {tc['args']['url'][:80]}...")
                    elif hasattr(msg, 'content') and msg.content:
                        elapsed = time.time() - segment_start
                        print(f"\n   [{idx}] 💭 Agent thinking... (at {elapsed:.1f}s)")
                        content_preview = str(msg.content)[:120]
                        print(f"       {content_preview}{'...' if len(str(msg.content)) > 120 else ''}")
                elif msg.type == 'tool':
                    # Extract credits from tool result
                    try:
                        tool_result = json.loads(msg.content)
                        credits = tool_result.get('credits_used', 0)
                        exec_time = tool_result.get('execution_time_seconds', 0)
                        segment_credits += credits
                        elapsed = time.time() - segment_start
                        success = tool_result.get('success', True)
                        
                        # Track search results
                        if tool_result.get('query'):  # This is a search result
                            search_results_log.append({
                                'query': tool_result.get('query'),
                                'results_count': tool_result.get('results_count', 0),
                                'results': tool_result.get('results', []),
                                'credits_used': credits,
                                'execution_time_seconds': exec_time
                            })
                        
                        if credits == 0:
                            print(f"   [{idx}] ⚠️  Tool returned 0 credits (likely failed/empty) | Segment at {elapsed:.1f}s")
                            note = tool_result.get('note', '')
                            if note:
                                print(f"       Note: {note[:80]}")
                        else:
                            print(f"   [{idx}] ✅ Tool completed in {exec_time:.2f}s (Credits: {credits}) | Segment at {elapsed:.1f}s")
                    except:
                        elapsed = time.time() - segment_start
                        print(f"   [{idx}] ✅ Tool result received | Segment at {elapsed:.1f}s")
        
        final_message = result["messages"][-1].content
        tool_calls = len([m for m in result['messages'] if m.type == 'tool'])
        
        # Parse and display structured market sizing
        print(f"\n{'='*80}")
        print(f"📊 STRUCTURED RESULTS - {segment_name}")
        print(f"{'='*80}")
        try:
            # Extract JSON from markdown code blocks if present
            content_text = final_message
            if isinstance(content_text, list) and len(content_text) > 0:
                content_text = content_text[0].get('text', '')
            
            if '```json' in content_text:
                json_start = content_text.find('```json') + 7
                json_end = content_text.find('```', json_start)
                content_text = content_text[json_start:json_end].strip()
            
            market_data = json.loads(content_text)
            
            # Display population data with sources
            if 'struggle_aware_population' in market_data:
                pop = market_data['struggle_aware_population']
                print(f"\n👥 Population:")
                print(f"   Total Addressable: {pop.get('total_addressable', 0):,}")
                if 'total_addressable_source' in pop:
                    print(f"   📚 Source: {pop['total_addressable_source'][:60]}...")
                print(f"   Struggle-Aware: {pop.get('final_struggle_aware_count', 0):,}")
                if 'filters' in pop:
                    print(f"   Filters Applied: {len(pop['filters'])}")
                    for f in pop['filters'][:2]:  # Show first 2 filters with sources
                        if 'source' in f:
                            print(f"      • {f.get('type', 'Filter')}: {f.get('percentage', 0)}% ({f['source'][:50]}...)")
            
            # Display pricing scenarios with comparables
            if 'pricing_scenarios' in market_data:
                scenarios = market_data['pricing_scenarios']
                print(f"\n💰 Pricing Scenarios:")
                for scenario in scenarios:
                    tier = scenario.get('tier', 'Unknown')
                    annual = scenario.get('annual_price_usd', 0)
                    monthly = scenario.get('monthly_price_usd', 0)
                    som = scenario.get('som_year_1_revenue_usd', 0)
                    print(f"   {tier}: ${monthly}/mo (${annual}/yr) → Year 1 SOM: ${som:,.0f}")
                    # Show comparables if available
                    if 'comparable_solutions' in scenario and scenario['comparable_solutions']:
                        comps = scenario['comparable_solutions']
                        print(f"      📊 Comparables: {', '.join([c if isinstance(c, str) else c.get('solution', 'N/A') for c in comps[:2]])}")
        except Exception as e:
            print(f"⚠️  Could not parse structured output: {str(e)[:100]}")
            print(f"Raw output length: {len(str(final_message))} characters")
        
        segment_duration = time.time() - segment_start
        
        print(f"\n{'='*80}")
        print(f"✅ Completed: {segment_name}")
        print(f"   ⏱️  Time: {segment_duration:.1f}s ({segment_duration/60:.1f} min)")
        print(f"   🔍 Tool Usage: {segment_search_count} searches + {segment_scrape_count} scrapes = {segment_search_count + segment_scrape_count} total")
        print(f"   💰 Credits: {segment_credits} (${segment_credits * 0.03:.2f})")
        print(f"   🤖 Tokens: {segment_input_tokens:,} input + {segment_output_tokens:,} output = {segment_input_tokens + segment_output_tokens:,} total")
        print(f"{'='*80}\n")
        
        return {
            "segment_name": segment_name,
            "priority": priority,
            "structured_data": market_data if 'market_data' in locals() else None,
            "raw_result": final_message,
            "tool_calls": tool_calls,
            "search_count": segment_search_count,
            "scrape_count": segment_scrape_count,
            "credits_used": segment_credits,
            "input_tokens": segment_input_tokens,
            "output_tokens": segment_output_tokens,
            "total_tokens": segment_input_tokens + segment_output_tokens,
            "duration_seconds": round(segment_duration, 2),
            "timestamp": datetime.now().isoformat(),
            "search_results": search_results_log
        }
    
    # Process segments in parallel
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(process_segment, (i, seg)): seg for i, seg in enumerate(top_segments, 1)}
        
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                segment = futures[future]
                print(f"❌ Error processing {segment.get('segment_name', 'Unknown')}: {str(e)[:100]}")
    
    phase2_duration = time.time() - phase2_start
    
    # Collect all search results from all segments
    all_search_results = []
    for result in results:
        if 'search_results' in result:
            for search_result in result['search_results']:
                search_result['segment'] = result['segment_name']  # Tag with segment name
            all_search_results.extend(result['search_results'])
    
    total_phase2_credits = sum(r.get('credits_used', 0) for r in results)
    total_phase2_searches = sum(r.get('search_count', 0) for r in results)
    total_phase2_scrapes = sum(r.get('scrape_count', 0) for r in results)
    total_phase2_input_tokens = sum(r.get('input_tokens', 0) for r in results)
    total_phase2_output_tokens = sum(r.get('output_tokens', 0) for r in results)
    total_phase2_tokens = total_phase2_input_tokens + total_phase2_output_tokens
    
    avg_segment_time = phase2_duration / len(results) if results else 0
    parallel_speedup = (sum(r.get('duration_seconds', 0) for r in results) / phase2_duration) if phase2_duration > 0 else 1
    
    print("\n" + "="*80)
    print("✅ PHASE 2 COMPLETED!")
    print(f"   ⏱️  Time: {phase2_duration:.1f}s ({phase2_duration/60:.1f} min) - Avg {avg_segment_time:.1f}s per segment")
    print(f"   🚀 Parallel speedup: {parallel_speedup:.1f}x (vs sequential)")
    print(f"   Market sized: {len(results)} segments")
    print(f"   🔍 Tool Usage: {total_phase2_searches} searches + {total_phase2_scrapes} scrapes = {total_phase2_searches + total_phase2_scrapes} total")
    print(f"   💰 Credits: {total_phase2_credits} (${total_phase2_credits * 0.03:.2f})")
    print(f"   🤖 Tokens: {total_phase2_input_tokens:,} input + {total_phase2_output_tokens:,} output = {total_phase2_tokens:,} total")
    print("="*80 + "\n")
    
    return {
        "results": results,
        "phase2_duration": phase2_duration,
        "total_credits": total_phase2_credits,
        "total_searches": total_phase2_searches,
        "total_scrapes": total_phase2_scrapes,
        "total_input_tokens": total_phase2_input_tokens,
        "total_output_tokens": total_phase2_output_tokens,
        "search_results": all_search_results
    }


def run_complete_pipeline(idea: str, jtbd: str, location: str = "United States"):
    """
    Run the complete two-phase pipeline:
    PHASE 1: Generate 8-12 segments with priorities
    PHASE 2: Market sizing for top 3 (PRIMARY + 2 SECONDARY)
    
    Args:
        idea: Product/service idea (string or dict)
        jtbd: Jobs to be done (string or dict)
        location: Geographic location
        
    Returns:
        Complete pipeline results with both phases
    """
    
    pipeline_start = time.time()
    
    print("\n" + "="*80)
    print("🚀 COMPLETE LANGCHAIN + FIRECRAWL PIPELINE")
    print("="*80)
    print(f"Idea: {idea if isinstance(idea, str) else idea.get('solution', idea)}")
    print(f"Location: {location}")
    print(f"\nEstimated cost: $0.96-1.20 (vs $29-30 with Firecrawl Agent API)")
    print("="*80 + "\n")
    
    # Extract idea string if dict
    if isinstance(idea, dict):
        idea_str = idea.get('solution', idea.get('problem', str(idea)))
    else:
        idea_str = idea
    
    # PHASE 1: Generate segments
    segment_result = generate_segments(idea_str, jtbd, location)
    
    # TODO: Parse segment_result to extract segments list
    # For now, create mock segments for testing
    mock_segments = [
        {"segment_name": "Segment 1", "priority_level": "primary", "description": "Primary segment"},
        {"segment_name": "Segment 2", "priority_level": "secondary", "description": "Secondary 1"},
        {"segment_name": "Segment 3", "priority_level": "secondary", "description": "Secondary 2"},
    ]
    
    # PHASE 2: Market sizing for top 3
    sizing_response = size_market_for_segments(idea_str, jtbd, mock_segments, location)
    sizing_results = sizing_response["results"]
    
    # Calculate total credits, tokens, and tool usage
    phase1_credits = segment_result.get('credits_used', 0)
    phase2_credits = sizing_response["total_credits"]
    total_credits = phase1_credits + phase2_credits
    total_cost = total_credits * 0.03
    
    phase1_searches = segment_result.get('search_count', 0)
    phase1_scrapes = segment_result.get('scrape_count', 0)
    phase2_searches = sizing_response["total_searches"]
    phase2_scrapes = sizing_response["total_scrapes"]
    total_searches = phase1_searches + phase2_searches
    total_scrapes = phase1_scrapes + phase2_scrapes
    
    phase1_input_tokens = segment_result.get('input_tokens', 0)
    phase1_output_tokens = segment_result.get('output_tokens', 0)
    phase1_total_tokens = phase1_input_tokens + phase1_output_tokens
    
    phase2_input_tokens = sizing_response["total_input_tokens"]
    phase2_output_tokens = sizing_response["total_output_tokens"]
    phase2_total_tokens = phase2_input_tokens + phase2_output_tokens
    
    total_input_tokens = phase1_input_tokens + phase2_input_tokens
    total_output_tokens = phase1_output_tokens + phase2_output_tokens
    total_tokens = total_input_tokens + total_output_tokens
    
    pipeline_duration = time.time() - pipeline_start
    phase1_duration = segment_result.get('duration_seconds', 0)
    phase2_duration = sizing_response["phase2_duration"]
    
    # Combine results
    complete_results = {
        "metadata": {
            "idea": idea_str,
            "jtbd": jtbd,
            "location": location,
            "model": "gemini-3-flash-preview",
            "timestamp": datetime.now().isoformat()
        },
        "phase1_segments": segment_result,
        "phase2_market_sizing": sizing_results,
        "all_search_results": {
            "phase1": segment_result.get('search_results', []),
            "phase2": sizing_response.get('search_results', [])
        },
        "tool_usage_summary": {
            "phase1": {
                "searches": phase1_searches,
                "scrapes": phase1_scrapes,
                "total_tools": phase1_searches + phase1_scrapes
            },
            "phase2": {
                "searches": phase2_searches,
                "scrapes": phase2_scrapes,
                "total_tools": phase2_searches + phase2_scrapes
            },
            "total": {
                "searches": total_searches,
                "scrapes": total_scrapes,
                "total_tools": total_searches + total_scrapes
            }
        },
        "cost_summary": {
            "phase1_credits": phase1_credits,
            "phase2_credits": phase2_credits,
            "total_credits": total_credits,
            "total_cost_usd": round(total_cost, 2),
            "cost_per_credit": 0.03
        },
        "token_summary": {
            "phase1": {
                "input_tokens": phase1_input_tokens,
                "output_tokens": phase1_output_tokens,
                "total_tokens": phase1_total_tokens
            },
            "phase2": {
                "input_tokens": phase2_input_tokens,
                "output_tokens": phase2_output_tokens,
                "total_tokens": phase2_total_tokens
            },
            "total": {
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "total_tokens": total_tokens
            }
        },
        "timing_summary": {
            "phase1_seconds": phase1_duration,
            "phase2_seconds": phase2_duration,
            "total_seconds": round(pipeline_duration, 2),
            "total_minutes": round(pipeline_duration / 60, 2)
        }
    }
    
    # Save results
    output_file = f"langchain_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(complete_results, f, indent=2)
    
    print("\n" + "="*80)
    print("🎉 PIPELINE COMPLETE!")
    print("="*80)
    
    print(f"\n⏱️  TOTAL TIME:")
    print(f"   Phase 1 (Segments):      {phase1_duration:6.1f}s ({phase1_duration/60:4.1f} min)")
    print(f"   Phase 2 (Market Sizing): {phase2_duration:6.1f}s ({phase2_duration/60:4.1f} min)")
    print(f"   {'-'*60}")
    print(f"   ⏱️  TOTAL:                {pipeline_duration:6.1f}s ({pipeline_duration/60:4.1f} min)")
    
    print(f"\n🔍 FIRECRAWL TOOL USAGE:")
    print(f"   Phase 1 (Segments):      {phase1_searches:2d} searches + {phase1_scrapes:2d} scrapes = {phase1_searches + phase1_scrapes:2d} tools")
    print(f"   Phase 2 (Market Sizing): {phase2_searches:2d} searches + {phase2_scrapes:2d} scrapes = {phase2_searches + phase2_scrapes:2d} tools")
    print(f"   {'-'*60}")
    print(f"   📊 TOTAL:                {total_searches:2d} searches + {total_scrapes:2d} scrapes = {total_searches + total_scrapes:2d} tools")
    
    print(f"\n💰 FIRECRAWL COST BREAKDOWN:")
    print(f"   Phase 1 (Segments):      {phase1_credits:3d} credits  ${phase1_credits * 0.03:6.2f}")
    print(f"   Phase 2 (Market Sizing): {phase2_credits:3d} credits  ${phase2_credits * 0.03:6.2f}")
    print(f"   {'-'*50}")
    print(f"   💵 TOTAL:                {total_credits:3d} credits  ${total_cost:6.2f}")
    print(f"\n   🎯 Target: 32-40 credits ($0.96-1.20)")
    print(f"   📈 vs Agent API: ~97 credits ($2.91) → Saved ${2.91 - total_cost:.2f}!")
    
    print(f"\n🤖 GEMINI 3 FLASH TOKEN USAGE:")
    print(f"   Phase 1 (Segments):      {phase1_input_tokens:6,} in + {phase1_output_tokens:6,} out = {phase1_total_tokens:7,} total")
    print(f"   Phase 2 (Market Sizing): {phase2_input_tokens:6,} in + {phase2_output_tokens:6,} out = {phase2_total_tokens:7,} total")
    print(f"   {'-'*70}")
    print(f"   📊 TOTAL:                {total_input_tokens:6,} in + {total_output_tokens:6,} out = {total_tokens:7,} total")
    print(f"\n   Model: gemini-3-flash-preview")
    print(f"   Avg tokens per tool: {total_tokens // (total_searches + total_scrapes) if (total_searches + total_scrapes) > 0 else 0:,}")
    
    print("\n" + "="*80)
    print(f"💾 Complete results saved to: {output_file}\n")
    
    return complete_results


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Test case - Gym Trainer Marketplace
    JTBD = {
        "narrative": "When I am juggling kids, work, and errands and finally see a small gap in my calendar, help me instantly match with a trainer and unlock a gym without sending a single text, so that I can squeeze in a workout and feel productive before the window closes.",
        "business_logic": {
            "company_type": "Marketplace (Uber-style)",
            "key_metric": "Match Rate / Time-to-Book",
            "model": "Transaction Fee (Cut of the booking)"
        },
        "founders_path": {
            "builder_dna": "The Engineer. Your focus is on algorithms, liquidity, and reducing click-count to zero.",
            "hard_part": "Liquidity Crisis. You need enough trainers and gyms in one zip code to guarantee an instant match, or users will churn immediately.",
            "user_vibe": "Invisible & Fast"
        }
    }
    
    IDEA = {
        "problem": "Scheduling friction and lack of access to local facilities making it hard for suburban parents and trainers to connect for short sessions",
        "target_user": "Suburban parents and independent personal trainers",
        "solution": "A mobile app that automates 30-minute scheduling, micro-gym booking, and smart lock access"
    }
    
    LOCATION = "United States"
    
    # Run complete pipeline
    result = run_complete_pipeline(
        idea=IDEA,
        jtbd=JTBD,
        location=LOCATION
    )
    
    print("\n✅ Pipeline completed!")
    print(f"📊 Check output file for full results")
