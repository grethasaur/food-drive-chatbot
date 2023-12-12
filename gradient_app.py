# Import necessary libraries
import streamlit as st
from gradientai import Gradient

def main():
    # Streamlit title and description
    st.title("Interactive Food Drive Assistant")
    st.write("Ask a question about the Food Drive!")

    with Gradient() as gradient:
        base_model = gradient.get_base_model(base_model_slug="nous-hermes2")
        new_model_adapter = base_model.create_model_adapter(name="interactive_food_drive_model")


        user_input = st.text_input("Ask your question:")
        if user_input and user_input.lower() not in ['quit', 'exit']:
            samples = [
            { "inputs": "### Instruction: Provide information about the food drive event. \n\n### Response: The food drive event is a community initiative aimed at collecting food items for those in need. We encourage participants to donate non-perishable food items such as canned goods, rice, pasta, and other essential items. Your contributions will make a difference in the lives of individuals and families facing food insecurity." },
            { "inputs": "### Instruction: What is the purpose of the food drive campaign? \n\n### Response: The food drive campaign aims to gather food donations to support local communities in need. By donating non-perishable items, you are helping to alleviate hunger and improve the lives of those facing food shortages. Your generous contributions are vital in making this initiative successful." },
            { "inputs": "### Instruction: Tell me more about how the food drive works. \n\n### Response: The food drive operates by encouraging individuals and organizations to donate non-perishable food items. These items can be dropped off at designated collection points or picked up by volunteers on specified dates. All collected food items are then distributed to local food banks, shelters, and needy families. Your participation in the food drive will contribute to ensuring that everyone in our community has access to nutritious meals." },
            { "inputs": "### Instruction: How can I get involved in the food drive? \n\n### Response: Getting involved is easy! You can volunteer your time by helping with collections, spreading awareness, or even organizing your own donation event. Every effort counts toward supporting our community members in need." },
            { "inputs": "### Instruction: What impact will my donation have on the community? \n\n### Response: Your donation directly supports local families and individuals facing food insecurity. By contributing non-perishable items, you're providing essential sustenance and hope to those who need it most." },
            { "inputs": "### Instruction: Where can I drop off my donations for the food drive? \n\n### Response: You can drop off your donations at designated collection points across the city. Check our website or contact us for specific locations and hours. Your contributions will go a long way in helping our cause!" },
            { "inputs": "### Instruction: Can my company/organization participate in the food drive? \n\n### Response: Absolutely! We welcome corporate or organizational involvement. You can organize a donation drive within your workplace or collaborate with us to make a substantial impact on our community's well-being." },
            { "inputs": "### Instruction: What specific food items are most needed for donations? \n\n### Response: While all donations are appreciated, items like canned vegetables, protein-rich foods (canned meat, peanut butter), grains, and hygiene products are highly sought after. Your donations of these items will help meet varied nutritional needs." }
            ]
      
            #fine tuning
            num_epochs = 2
            count = 0
            while count < num_epochs:
              new_model_adapter.fine_tune(samples=samples)
              count = count + 1

            sample_query = f"### Instruction: {user_input} \n\n### Response:"
            st.markdown(f"Asking: {sample_query}")

            # after fine-tuning
            complete_response = new_model_adapter.complete(
                query=sample_query,
                max_generated_token_count=100)
            st.markdown(f"Generated: {complete_response.generated_output}")

        # Delete the model adapter after generating the response
        new_model_adapter.delete()

if __name__ == "__main__":
    main()
