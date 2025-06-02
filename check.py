import google.generativeai as genai
import asyncio

def configure_gemini(api_key):
    """Configure the Gemini API with the provided API key."""
    genai.configure(api_key=api_key)

async def get_gemini_response(prompt):
    try:
        # Initialize the model
        model = genai.GenerativeModel('gemini-1.5-pro')
        # Generate the response
        response = await asyncio.to_thread(
            lambda: model.generate_content(prompt).text
        )
        return response
    except Exception as e:
        print(f"Error querying Gemini: {e}")
        return "Error occurred"

async def get_crop_details_from_gemini(crop_name, api_key):
    configure_gemini(api_key)
    
    prompts = {
        "cultivation_guide": f"Provide a detailed cultivation guide for {crop_name}.",
        "soil_requirements": f"What are the soil requirements for cultivating {crop_name}?",
        "climate_conditions": f"What climate conditions are ideal for growing {crop_name}?",
        "harvesting": f"Explain the harvesting process for {crop_name}.",
        "diseases": f"What are common diseases affecting {crop_name} and how to manage them?",
        "market_value": f"Discuss the market value of {crop_name} and factors that affect it."
        "only provide this information not any other info if you didnt get any correct information just return no info found "
    }

    # Using asyncio.gather to query Gemini concurrently
    responses = await asyncio.gather(
        *[get_gemini_response(prompt) for prompt in prompts.values()]
    )
    
    crop_info = {
        key: response if response else "Information not available"
        for key, response in zip(prompts.keys(), responses)
    }
    
    return crop_info

if __name__ == "__main__":
    API_KEY = "AIzaSyCdQ2JM7Ww1cT-uEgFwq54SioZM4KGifHI"  # Replace with your actual API key
    crop_name = "Tomato"  # Example crop name
    
    try:
        crop_details = asyncio.run(get_crop_details_from_gemini(crop_name, API_KEY))
        print("Crop Details:")
        for key, value in crop_details.items():
            print(f"\n{key.upper()}:")
            print(value)
    except Exception as e:
        print(f"Error: {str(e)}")