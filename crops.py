def crop(crop_name):
    crop_data = {
    "wheat":["/static/images/wheat.jpg", "Karnataka, Punjab, Haryana, Rajasthan, M.P., Bihar", "Rabi (Sown in Winter & Harvested in the Spring)","Sri Lanka, United Arab Emirates, Taiwan"],
    "paddy":["/static/images/paddy.jpg", "Karnataka, U.P., Andhra Pradesh, Punjab, T.N.", "Kharif/Monsoon Crops","Bangladesh, Saudi Arabia, Iran"],
    "maize":["/static/images/maize.jpg", "Karnataka, Andhra Pradesh, Tamil Nadu, Rajasthan, Maharashtra", "Kharif/Monsoon Crops", "Hong Kong, United Arab Emirates, France"],
    "ragi":["/static/images/ragi.jpg",  "Karnataka,Maharashtra, Tamil Nadu and Uttarakhand", "Kharif/Monsoon Crops", "United Arab Emirates, New Zealand, Bahrain"],
    "sugarcane":["/static/images/sugarcane.jpg","Karnataka ,Uttar Pradesh, Maharashtra, Tamil Nadu, Andhra Pradesh" , "Kharif/Monsoon Crops", "Kenya, United Arab Emirates, United Kingdom"]
    }
    return crop_data[crop_name]