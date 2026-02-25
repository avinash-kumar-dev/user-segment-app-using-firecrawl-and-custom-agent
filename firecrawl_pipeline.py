"""
Firecrawl Complete Pipeline: User Segment Generation → Market Sizing
Generates 8-12 user segments, then performs market sizing analysis for each
"""

from firecrawl import FirecrawlApp
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
import json
from datetime import datetime
import time

# Initialize Firecrawl
FIRECRAWL_API_KEY = "fc-ee1747ed45c74785ab494d76f578d9a2"
app = FirecrawlApp(api_key=FIRECRAWL_API_KEY)

# ============================================================================
# PYDANTIC MODELS FOR SEGMENT GENERATION
# ============================================================================

class ValidationField(BaseModel):
    """Individual validation field with value and source URLs"""
    value: str = Field(description="The validated finding or evidence")
    source_urls: List[str] = Field(description="Array of exact URLs verifying this claim")

class ValidationData(BaseModel):
    """Comprehensive validation data for a segment"""
    behavioral_evidence: ValidationField = Field(description="Proof of segment behaviors with sources")
    pain_intensity: ValidationField = Field(description="Evidence of pain severity with sources")
    current_solutions: ValidationField = Field(description="What they use today and why it fails, with sources")
    segment_accessibility: ValidationField = Field(description="How easy to reach this segment, with sources")
    data_quality: Literal["high", "medium", "low"] = Field(description="Quality assessment based on source credibility")

class UserSegment(BaseModel):
    """A single user segment with comprehensive market data"""
    segment_name: str = Field(description="Short, memorable name")
    description: str = Field(description="Who they are and their context")
    context_circumstance: str = Field(description="When/where the struggle happens")
    key_constraints: List[str] = Field(description="List of blocking constraints")
    buyer_type: Literal["self-serve", "manager-approved", "procurement-driven"]
    struggle_frequency: Literal["daily", "weekly", "monthly", "occasional"]
    struggle_intensity: Literal["critical", "high", "medium", "low"]
    existing_alternatives: str = Field(description="What they use today")
    priority_level: Literal["primary", "secondary", "alternative"] = Field(description="Market entry priority")
    priority_rationale: str = Field(description="2-3 sentences explaining priority assignment")
    product_vibe: str = Field(description="Emotional/functional product personality")
    possible_features: List[str] = Field(description="5-8 features specific to this segment", min_length=5, max_length=8)
    validation_data: ValidationData

class SegmentGenerationResult(BaseModel):
    """Complete result of segment generation"""
    segments: List[UserSegment] = Field(description="List of 8-12 user segments", min_length=8, max_length=12)

# ============================================================================
# PYDANTIC MODELS FOR MARKET SIZING
# ============================================================================

class PopulationData(BaseModel):
    """Population and struggle-aware calculations"""
    total_population: int = Field(description="Total addressable population number")
    population_source: str = Field(description="Source citation for population data")
    prevalence_rate: float = Field(description="Decimal rate of those experiencing the struggle", ge=0, le=1)
    prevalence_source: str = Field(description="Source showing prevalence percentage")
    struggle_aware_count: int = Field(description="Calculated struggle-aware population")
    calculation_logic: str = Field(description="Step-by-step calculation explanation")
    confidence: Literal["High", "Medium", "Low"]

class ComparableSolution(BaseModel):
    """Competitor pricing information"""
    solution_name: str
    price: str = Field(description="Price with period (e.g., $90/year)")
    source: str = Field(description="URL or report name")

class PricingScenario(BaseModel):
    """Individual pricing tier analysis"""
    tier: str = Field(description="Low/Mid/High scenario name")
    annual_price: float = Field(description="Annual price in USD")
    local_price: Optional[str] = Field(description="Price in local currency if different")
    pricing_rationale: str = Field(description="4-6 sentences explaining price with benchmarks")
    comparable_solutions: List[ComparableSolution] = Field(min_length=2, max_length=5)
    sam_revenue: float = Field(description="Struggle-aware count × price")
    capture_rate: str = Field(description="Percentage like '2%'")
    som_year_1: float = Field(description="Year 1 revenue estimate")
    som_reasoning: str = Field(description="Why this capture rate is realistic")

class PricingAnalysis(BaseModel):
    """Complete pricing analysis with multiple scenarios"""
    pricing_scenarios: List[PricingScenario] = Field(min_length=3, max_length=3)
    recommended_scenario: Literal["Low", "Mid", "High"]
    recommendation_reasoning: str

class MarketSizingResult(BaseModel):
    """Complete market sizing analysis for one segment"""
    population: PopulationData
    pricing: PricingAnalysis

# ============================================================================
# CREDIT TRACKING
# ============================================================================

class CreditTracker:
    """Track Firecrawl credits, tokens, and costs"""
    def __init__(self):
        self.operations = []
        self.total_credits = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
    
    def add_operation(self, operation_name: str, credits_used: int, duration_seconds: float, 
                     input_tokens: int = 0, output_tokens: int = 0, metadata: dict = None):
        operation_data = {
            "operation": operation_name,
            "credits": credits_used,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "duration_seconds": round(duration_seconds, 2),
            "timestamp": datetime.now().isoformat()
        }
        if metadata:
            operation_data["metadata"] = metadata
        
        self.operations.append(operation_data)
        self.total_credits += credits_used
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
    
    def get_summary(self):
        total_time = sum(op["duration_seconds"] for op in self.operations)
        estimated_cost = self.total_credits * 0.03  # $30 per 1000 credits
        
        return {
            "total_credits": self.total_credits,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "estimated_cost_usd": round(estimated_cost, 2),
            "cost_per_credit": 0.03,
            "total_time_seconds": round(total_time, 2),
            "operations": self.operations
        }

# ============================================================================
# PROMPT TEMPLATES
# ============================================================================

def create_segment_generation_prompt(idea: str, jtbd: str, location: str) -> str:
    """Generate the segment generation prompt"""
    return f"""You are an Expert Product Strategist specializing in market opportunity analysis using the Jobs-to-be-Done (JTBD) framework.

Your mission is to help an early-stage entrepreneur identify **all potential user segments** ("The Who") for their startup idea, with comprehensive market data for each.

---

# INPUTS

**User's Startup Idea:**
{idea}

**Selected JTBD (Jobs-to-be-Done):**
{jtbd}

**Target Location/Market:** {location}

**JTBD Format Explained:**
The JTBD follows this structure: **[user_segment] in [situation] need to [JTBD] so that they can [outcome]**

This represents the **specific struggle** a particular user segment faces. Your job is to identify **variations of this user segment** - finding sub-groups who experience this struggle with different intensities, contexts, constraints, and willingness to pay.

---

# YOUR TASK: GENERATE 8-12 SEGMENTS WITH COMPREHENSIVE DATA

You must generate 8-12 distinct user segments who experience this specific struggle. **ALL segments should be analyzed equally with the same depth of research and data backing.**

**IMPORTANT: Geographic Context**
- All market data, research, and statistics should be relevant to {location}
- When searching for data, prioritize {location}-specific sources
- If global data is used, explicitly note it and contextualize for {location}

**Segment Count Guidelines**: Generate between 8-12 segments total (NOT exactly 10). Vary the count based on idea complexity - simpler ideas may warrant 8 segments, complex multi-sided markets may need 12.

## Prioritization Requirements (CRITICAL):

After identifying all 8-12 segments, you MUST assign priority levels:

1. **PRIMARY (1 segment only)**: The single most promising segment based on:
   - Market size and accessibility
   - Pain intensity and willingness to pay
   - Speed to revenue/traction
   - Strategic value (e.g., word-of-mouth, network effects)
   
2. **SECONDARY (2 segments only)**: The next two most promising segments that complement the primary

3. **ALTERNATIVE (all remaining segments)**: Other viable segments for future expansion

**For each segment, provide a priority_rationale** explaining why it received that priority level. The rationale should reference specific data points (market size, accessibility, pain metrics) that justify the prioritization.

## Research Requirements for EVERY Segment (CRITICAL):

**MANDATORY: Use Google Search grounding to find REAL DATA for each segment:**
- Cite actual market research reports showing market opportunity
- Reference real user studies/surveys showing pain intensity  
- Quote actual industry reports on market trends
- Link to competitor revenue/traction data
- Include specific numbers, percentages, dollar amounts
- **MUST include URLs/sources** for all data cited

**Example format**: "According to [Source Name] [Year] report (https://...), this segment shows [specific finding]. [Another Source] (https://...) found [additional data point]."

**Equal Research Depth**: All segments should receive equal research effort - no prioritization or ranking. Each segment deserves the same level of data validation and market analysis.

**Specificity Required**: No generic segments allowed. Each must have specific context, constraints, and alternatives that distinguish it from others.

## Variation Dimensions:

1. **Context/Circumstance**: *When* and *where* does the struggle happen? (e.g., during commute, weekend planning, quarterly reviews, crisis moments)
2. **Key Constraints**: What blocks them? (time scarcity, budget limits, compliance requirements, coordination complexity, technical skill gaps)
3. **Buyer Type**: Who makes the purchase decision? (individual self-serve, manager-approved, procurement committee)
4. **Struggle Frequency**: How often do they face this? (daily, weekly, monthly, occasional)
5. **Struggle Intensity**: How painful is it? (critical, high, medium, low)
6. **Existing Alternatives**: What do they use today? (Excel, manual process, competitor product, outsourced agency, "do nothing")

## Product Implications Analysis:

For each segment, determine:

1. **The Vibe**: What is the emotional/functional "personality" of the product for this segment?
   - Examples: "Premium concierge service", "No-frills speed tool", "Enterprise compliance guardian", "Social fitness game"

2. **Possible Features**: List 5-8 features that directly address their specific constraints
   - Example: If constraint is "time scarcity" → feature: "1-click booking", "auto-scheduling"
   - Example: If constraint is "budget limits" → feature: "pay-per-use pricing", "freemium tier"

3. **Market Data**: MUST include real data with sources/URLs:
   - Market opportunity size and metrics
   - Market trends and dynamics
   - Pain intensity evidence (surveys, studies)
   - Competitor landscape and traction
   - Adoption barriers
   - Willingness to pay data

---

# CRITICAL CONSTRAINT

**REAL DATA REQUIREMENT**: EVERY validation_data field (behavioral_evidence, pain_intensity, current_solutions, segment_accessibility) MUST include:
1. A specific value/finding
2. An array of source_urls (exact URLs from Google Search)

Do NOT use assumptions or estimates without citing sources. Each claim MUST have verifiable source URLs. This granular source tracking is critical for fact-checking and verification.

---

Generate the comprehensive segment analysis now with full market data and sources for each segment."""

def create_market_sizing_prompt(idea: str, jtbd: str, segment_name: str, segment_description: str, location: str) -> str:
    """Generate the market sizing prompt for a specific segment"""
    return f"""# Role
You are a Senior VC Analyst specializing in STRUGGLE-BASED MARKET SIZING with rigorous data sourcing.

Your goal: Calculate market size based on WHO ACTUALLY FEELS THE PROBLEM, not just demographics.

**Core Methodology:**  
- **Struggle-Aware SAM**: Only count people experiencing the struggle at a meaningful frequency
- **Location-Specific**: All data must be relevant to {location}
- **Source-Backed**: Every number must cite a verifiable source
- **Defendable Pricing**: Price must be justified by real market comparables and ROI

---

# INPUTS

**User Idea:** {idea}  
**JTBD Statement:** {jtbd}  
**Target Segment:** {segment_name}  
**Segment Description:** {segment_description}  
**Target Location:** {location}

---

# STEP 1: STRUGGLE-AWARE POPULATION SIZING

**CRITICAL**: You MUST calculate a "struggle-aware" population, NOT total market.

### Task:
1. **Start with Total Addressable Population**
   - Find the total number of [segment type] in {location}
   - Example: "15M freelancers in India (NITI Aayog 2024)"
   - **MUST cite source**

2. **Filter to Struggle-Aware Population**
   - Apply filters to get those who ACTUALLY experience this struggle:
     - Geographic filter (if needed)
     - Frequency filter (only those experiencing problem daily/weekly/monthly)
     - Intensity filter (those for whom this is a significant pain)
   - Example: "15M × 25% (in tier-1 cities) × 73% (experience trust barrier) = 2.7M struggle-aware"
   - **Show calculation logic**
   - **Cite sources for each filter percentage**

3. **Provide Evidence**
   - Prevalence rate (what % experience this struggle?)
   - Frequency data (how often does it happen?)
   - Sources for both population AND prevalence

---

# STEP 2: DEFENDABLE PRICING with REAL DATA

**CRITICAL**: Price must be DEFENDED with real market comparables from {location}.

### Task:
1. **Find Market Benchmarks**
   - Search for 3-5 comparable solutions that {segment_name} in {location} currently pays for
   - Get actual pricing data (cite sources: pricing pages, industry reports, reviews)
   - Example: "Freelancer.com verification badge: $90 (source: pricing page)"

2. **Calculate Value/ROI**
   - What's the DOLLAR VALUE of solving this problem in {location}?
   - Consider local salaries, cost of living, revenue potential
   - Example: "Average freelancer in India loses ₹40,000 ($500) annually from trust issues (Payoneer 2024)"

3. **Generate 3 Pricing Scenarios**
   - **Low**: Mass market, high volume
   - **Mid**: Value-based, ROI-justified
   - **High**: Premium, done-for-you

For EACH scenario, provide:
- **Monthly/Annual Price** (in local currency + USD)
- **Pricing Rationale**: 3-4 sentences explaining:
  * How it compares to market benchmarks (cite specific competitors)
  * ROI calculation (price as % of value created)
  * Why this price is reasonable for {location}
- **Comparable Solutions**: List 2-3 real competitors with their prices and sources
- **SAM Revenue**: struggle_aware_count × annual_price
- **SOM Estimate**: Realistic Year 1 capture rate with justification

---

# CRITICAL REQUIREMENTS

✅ **Every number must have a source**  
✅ **Use {location}-specific data**  
✅ **Calculate struggle-aware population (not total market)**  
✅ **Provide 3+ comparable solution prices with sources**  
✅ **Show ROI calculation for pricing**  
✅ **Be honest about data confidence (High/Medium/Low)**

❌ **Do NOT use total market (e.g., "all freelancers")**  
❌ **Do NOT make up numbers without sources**  
❌ **Do NOT ignore location-specific context**

---

Generate the analysis now."""

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_complete_pipeline(idea: str, jtbd: str, location: str = "United States", 
                          size_all_segments: bool = False):
    """
    Complete pipeline: Generate segments → Market sizing for PRIMARY + SECONDARY only
    
    Args:
        idea: Startup idea description
        jtbd: Jobs-to-be-Done statement
        location: Target geographic location
        size_all_segments: If True, size all segments. If False (default), only size primary + secondary
    """
    print("="*80)
    print("🚀 FIRECRAWL COMPLETE PIPELINE")
    print("="*80)
    print(f"Idea: {idea}")
    print(f"JTBD: {jtbd}")
    print(f"Location: {location}\n")
    
    tracker = CreditTracker()
    pipeline_start = time.time()
    
    # ========================================================================
    # STEP 1: GENERATE USER SEGMENTS
    # ========================================================================
    print("="*80)
    print("STEP 1: GENERATING USER SEGMENTS (8-12 segments with priorities)")
    print("="*80)
    
    segment_prompt = create_segment_generation_prompt(idea, jtbd, location)
    
    print("🔍 Calling Firecrawl Agent for segment generation...")
    print("📝 This will search for market data, competitor info, and user studies...")
    
    segment_start = time.time()
    
    try:
        segment_result = app.agent(
            prompt=segment_prompt,
            schema=SegmentGenerationResult,
            model="spark-1-mini"
        )
        
        segment_duration = time.time() - segment_start
        
        # Extract credit usage (tokens not available from Firecrawl API)
        credits = segment_result.credits_used or 0
        
        # Track the operation
        tracker.add_operation(
            "Segment Generation", 
            credits,
            segment_duration,
            input_tokens=0,  # Not provided by Firecrawl API
            output_tokens=0,  # Not provided by Firecrawl API
            metadata={
                "model": "spark-1-mini",
                "segments_generated": len(segment_result.data.get("segments", [])),
                "note": "Token counts not available from Firecrawl API"
            }
        )
        
        segments = segment_result.data["segments"]
        print(f"\n✅ Generated {len(segments)} segments in {segment_duration:.1f}s")
        print(f"💰 Credits used: {credits}")
        if credits == 0:
            print(f"   ℹ️  Note: 0 credits may indicate free tier or test mode")
        
        # Display segments grouped by priority
        print("\n📊 SEGMENTS GENERATED:")
        
        primary_segments = [s for s in segments if s.get('priority_level') == 'primary']
        secondary_segments = [s for s in segments if s.get('priority_level') == 'secondary']
        alternative_segments = [s for s in segments if s.get('priority_level') == 'alternative']
        
        print(f"\n🎯 PRIMARY (1 segment):")
        for seg in primary_segments:
            print(f"   • {seg['segment_name']}")
            print(f"     {seg['description'][:80]}...")
            print(f"     Rationale: {seg.get('priority_rationale', 'N/A')[:100]}...")
        
        print(f"\n🎯 SECONDARY ({len(secondary_segments)} segments):")
        for seg in secondary_segments:
            print(f"   • {seg['segment_name']}")
            print(f"     {seg['description'][:80]}...")
        
        print(f"\n⚪ ALTERNATIVE ({len(alternative_segments)} segments):")
        for seg in alternative_segments:
            print(f"   • {seg['segment_name']}")
            
    except Exception as e:
        print(f"\n❌ Segment generation failed: {str(e)}")
        return None
    
    # ========================================================================
    # STEP 2: MARKET SIZING FOR PRIMARY + SECONDARY SEGMENTS ONLY
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 2: MARKET SIZING (PRIMARY + SECONDARY SEGMENTS)")
    print("="*80)
    
    # Filter to PRIMARY + SECONDARY segments only
    if size_all_segments:
        segments_to_process = segments
        print(f"⚠️ Sizing ALL {len(segments)} segments (custom mode)")
    else:
        segments_to_process = [s for s in segments if s.get('priority_level') in ['primary', 'secondary']]
        print(f"🎯 Sizing {len(segments_to_process)} high-priority segments (1 PRIMARY + 2 SECONDARY)")
        print(f"⏭️ Skipping {len(segments) - len(segments_to_process)} ALTERNATIVE segments")
    
    market_sizing_results = []
    
    for i, segment in enumerate(segments_to_process, 1):
        priority_emoji = "🎯" if segment.get('priority_level') == 'primary' else "🔸"
        print(f"\n{'='*80}")
        print(f"SEGMENT {i}/{len(segments_to_process)}: {segment['segment_name']} {priority_emoji} [{segment.get('priority_level', 'N/A').upper()}]")
        print(f"{'='*80}")
        
        sizing_prompt = create_market_sizing_prompt(
            idea=idea,
            jtbd=jtbd,
            segment_name=segment['segment_name'],
            segment_description=segment['description'],
            location=location
        )
        
        print(f"🔍 Analyzing market size for: {segment['segment_name'][:60]}...")
        
        sizing_start = time.time()
        
        try:
            sizing_result = app.agent(
                prompt=sizing_prompt,
                schema=MarketSizingResult,
                model="spark-1-mini"
            )
            
            sizing_duration = time.time() - sizing_start
            
            # Extract credit usage (tokens not available from Firecrawl API)
            credits = sizing_result.credits_used or 0
            
            # Track the operation
            tracker.add_operation(
                f"Market Sizing: {segment['segment_name']}", 
                credits,
                sizing_duration,
                input_tokens=0,  # Not provided by Firecrawl API
                output_tokens=0,  # Not provided by Firecrawl API
                metadata={
                    "model": "spark-1-mini",
                    "segment": segment['segment_name'],
                    "priority": segment.get('priority_level', 'unknown')
                }
            )
            
            market_data = sizing_result.data
            
            print(f"✅ Completed in {sizing_duration:.1f}s")
            print(f"💰 Credits: {credits}")
            print(f"📊 Struggle-aware population: {market_data['population']['struggle_aware_count']:,}")
            print(f"💵 Recommended: {market_data['pricing']['recommended_scenario']} tier")
            
            # Combine segment + market sizing
            market_sizing_results.append({
                "segment": segment,
                "market_sizing": market_data,
                "metadata": {
                    "credits_used": sizing_result.credits_used,
                    "duration_seconds": round(sizing_duration, 2)
                }
            })
            
            # Rate limiting to avoid overwhelming the API
            if i < len(segments_to_process):
                print("\n⏳ Waiting 2 seconds before next segment...")
                time.sleep(2)
                
        except Exception as e:
            print(f"❌ Market sizing failed: {str(e)}")
            market_sizing_results.append({
                "segment": segment,
                "market_sizing": None,
                "error": str(e)
            })
    
    # ========================================================================
    # FINAL RESULTS
    # ========================================================================
    pipeline_duration = time.time() - pipeline_start
    
    print("\n" + "="*80)
    print("📊 PIPELINE COMPLETE")
    print("="*80)
    
    credit_summary = tracker.get_summary()
    
    print(f"\n⏱️ TIMING:")
    print(f"   Total duration: {pipeline_duration:.1f}s ({pipeline_duration/60:.1f} minutes)")
    print(f"   Segment generation: {tracker.operations[0]['duration_seconds']:.1f}s")
    print(f"   Market sizing average: {sum(op['duration_seconds'] for op in tracker.operations[1:]) / len(tracker.operations[1:]):.1f}s per segment")
    
    print(f"\n💰 COST BREAKDOWN:")
    print(f"   Total credits: {credit_summary['total_credits']}")
    print(f"   Estimated cost: ${credit_summary['estimated_cost_usd']:.2f} (at $30 per 1,000 credits)")
    if credit_summary['total_credits'] == 0:
        print(f"   ℹ️  Note: 0 credits may indicate free tier, test mode, or included allowance")
    print(f"   Cost per credit: ${credit_summary['cost_per_credit']:.3f}")
    print(f"\n⚠️  TOKEN TRACKING:")
    print(f"   Firecrawl API does not provide token-level metrics")
    print(f"   Only credit usage is available for cost estimation")
    print(f"\n📊 PER OPERATION:")
    print(f"   Segment generation: {tracker.operations[0]['credits']} credits ({tracker.operations[0]['duration_seconds']:.1f}s)")
    if len(tracker.operations) > 1:
        market_sizing_ops = tracker.operations[1:]
        total_credits = sum(op['credits'] for op in market_sizing_ops)
        avg_credits = total_credits / len(market_sizing_ops)
        avg_time = sum(op['duration_seconds'] for op in market_sizing_ops) / len(market_sizing_ops)
        print(f"   Market sizing total: {total_credits} credits")
        print(f"   Market sizing (avg): {avg_credits:.1f} credits, {avg_time:.1f}s per segment")
    
    print(f"\n📈 RESULTS:")
    print(f"   Segments generated: {len(segments)}")
    print(f"   Segments sized: {len(market_sizing_results)}")
    print(f"   Success rate: {len([r for r in market_sizing_results if r.get('market_sizing')])} / {len(market_sizing_results)}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"pipeline_results_{timestamp}.json"
    
    output_data = {
        "metadata": {
            "idea": idea,
            "jtbd": jtbd,
            "location": location,
            "timestamp": datetime.now().isoformat(),
            "pipeline_duration_seconds": round(pipeline_duration, 2)
        },
        "credit_summary": credit_summary,
        "segments_generated": segments,
        "market_sizing_results": market_sizing_results
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n📁 Results saved: {output_file}")
    print("="*80)
    
    return output_data

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example inputs - Gym Trainer Marketplace
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
    
    # Run the complete pipeline
    # By default, only sizes PRIMARY (1) + SECONDARY (2) segments
    # Set size_all_segments=True to process all segments
    results = run_complete_pipeline(
        idea=IDEA,
        jtbd=JTBD,
        location=LOCATION,
        size_all_segments=False  # Only process PRIMARY + SECONDARY (recommended)
    )
    
    if results:
        print("\n✅ Pipeline completed successfully!")
        print(f"📊 Total segments: {len(results['segments_generated'])}")
        print(f"💰 Total cost: ${results['credit_summary']['estimated_cost_usd']:.2f}")
