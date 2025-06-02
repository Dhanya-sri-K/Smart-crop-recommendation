import google.generativeai as genai
import asyncio

def configure_gemini(api_key):
    """Configure the Gemini API with the provided API key."""
    genai.configure(api_key=api_key)

# Add retry logic with exponential backoff
async def get_gemini_response(prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = await asyncio.to_thread(
                lambda: model.generate_content(prompt).text
            )
            return response
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 1  # Exponential backoff
                print(f"Rate limit hit, waiting {wait_time} seconds...")
                await asyncio.sleep(wait_time)
                continue
            print(f"Error querying Gemini: {e}")
            return "Error occurred"

async def scrape_crop_details(crop_name):
    """Get crop details from Gemini API."""
    API_KEY = "AIzaSyB1TA_SYxKVoc1XRYl_rL7tsHL9B79Koks"  # Replace with your actual API key
    configure_gemini(API_KEY)
    
    prompts = {
        "cultivation_guide": f"""Provide a simple step-by-step guide for growing {crop_name}. Format as:
1. Land Preparation: (4-5 points)
2. Sowing: (2-3 points)
3. Care Instructions: (3-4 key points)
Keep each point brief and use simple language.""",

        "soil_requirements": f"""What type of soil is best for {crop_name}? Include:
• Best soil type
• pH level
• Drainage needs
Maximum 6-7 lines in simple language.""",

        "climate_conditions": f"""What weather conditions does {crop_name} need? Include:
• Temperature range to maintain
• Rainfall needs or irrigation amount
• Best growing season
Keep it to 3-4 lines.""",

        "harvesting": f"""Simple guide for harvesting {crop_name}:
• When to harvest (clear signs)
• How to harvest
• Expected yield
Maximum 4-5 lines.""",

        "diseases": f"""List 3 main diseases/pests affecting {crop_name}:
1. [Disease name]: Simple prevention/cure
2. [Disease name]: Simple prevention/cure
3. [Disease name]: Simple prevention/cure
Use basic terms farmers understand.""",

        "market_value": f"""Brief overview of {crop_name} market value:
• Current market rate
• Best time to sell
• Storage tips
Maximum 4 lines in simple language and it should be in indian ruppees."""
    }
    
    try:
        responses = await asyncio.gather(
            *[get_gemini_response(prompt) for prompt in prompts.values()]
        )
        
        crop_info = {
            key: response if response != "Error occurred" else "Information not available"
            for key, response in zip(prompts.keys(), responses)
        }
        
        return crop_info
    except Exception as e:
        print(f"Error getting crop details: {e}")
        return {key: "Information not available" for key in prompts.keys()}