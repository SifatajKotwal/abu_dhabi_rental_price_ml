# Abu Dhabi Rental Price Prediction: A Machine Learning Approach

>A structured data analysis project exploring how property characteristics & neighborhood dynamics influence rental pricing in Abu Dhabi

---

## Why This Project?

I started this project because I was curious about a market I kept hearing about but had never experienced firsthand. The UAE, & Abu Dhabi specifically, always came up in discussions about rapid urban development, luxury real estate & unique city planning. I wondered- how does rental pricing actually work in a place where everything seems to be built in the last 20 years? Does the logic I've seen in Indian metros where location premiums often override everything else ,apply there too?

I found a dataset of UAE property listings & decided to find out. This turned into a full exploration of Abu Dhabi's rental market cleaning messy real-world data, engineering features that capture livability rather than just size & testing which modeling approaches actually handle this kind of pricing complexity

Working on data from a place I've never lived forced me to research neighborhoods, understand the local property type classifications & think carefully about what features would actually matter to renters there. That distance became an advantage, I had to justify every assumption rather than relying on intuition

---

## What I Built

A complete ML pipeline that predicts annual rental prices for residential properties in Abu Dhabi using structural features (beds, baths, area) & location-based information. The project compares multiple modeling strategies to find what works best for structured tabular data with strong spatial & non-linear effects

---
### Key Components

**1. Data Cleaning**
- Filtered 73K+ UAE property records down to 23K Abu Dhabi listings  
- Removed extreme outliers (rent > 2M AED)  
- Handled missing coordinates & inconsistent categorical labels  

**2. Feature Engineering**
- Created `Total_Rooms` and `Room_Density` as livability proxies  
- Standardized furnishing labels  
- Grouped rare property types to prevent overfitting  

**3. Location Encoding**
- Implemented smoothed target encoding for 85 neighborhoods  
- Prevented leakage using smoothing & minimum sample thresholds  

**4. Model Comparison**
- Compared Linear Regression, Ridge, Random Forest, Gradient Boosting, Segmented RF and Neural Network (MLP)  

**5. Error Analysis**
- Identified higher error in top 10% luxury segment  
- Tested segmented modeling approach  

---

## The Data

**Source:** Dubai Real Estate Goldmine dataset (Dr. Murat ALTUN), filtered to ABU DHABI only

**Final dataset:** 21,075 properties after cleaning

| Feature                  | Description 
|--------------------------|------------------------------------------------------------
| `Beds` / `Baths`         | Basic structural capacity 
| `Area_in_sqft`           | Property size 
| `Total_Rooms`            | Engineered: Beds + 0.5*Baths (usability approximation) 
| `Room_Density`           | Engineered: Total_Rooms / Area (crowding/luxury indicator) 
| `Location_TE`            | Target-encoded neighborhood (smoothed to prevent overfitting) 
| `Type_grouped`           | Property type (Apartment/Villa/Townhouse/Other) 
| `Furnishing_clean`       | Standardized furnishing status 
| `Age_of_listing_in_days` | Temporal feature capturing listing lifecycle 

---

## What I Found

### Rental pricing is deeply non-linear

The relationship between size & rent isn't proportional. I binned properties by area & plotted median rents..the curve shows rapid increases for smaller units then diminishing returns. This immediately suggested linear models would struggle & it matched what I'd read about Abu Dhabi's market- smaller, well-located apartments often command disproportionate premiums compared to sprawling villas in outer areas

### Location matters a lot

Neighborhood level effects play a significant role in rental pricing. The target encoded location feature ranked among the most influential predictors in the ensemble models, indicating that pricing differences across neighborhoods are substantial
Even when structural features such as area, bedrooms, & bathrooms are included, location continues to contribute meaningful predictive power. This suggests that rental valuation in Abu Dhabi is not determined by property characteristics alone but also by localized demand dynamics
The smoothing strategy applied during target encoding helped capture these neighborhood level differences while reducing overfitting from high cardinality categories

### Tree ensembles beat everything else

| Model                | RMSE (AED) |MAE (AED) | R² Score 
|----------------------|------------|----------|------------             
| **Random Forest**    | **42,046** |**18,330**|  **0.886**  
| Gradient Boosting    | 43,812     |  21,628  |  0.857 
| Neural Network (MLP) | 51,013     |  25,728  |  0.793 
| Ridge Regression     | 58,852     |  30,847  |  0.731 
| Linear Regression    | 58,854     |  30,847  |  0.731 

The Random Forest's ability to capture feature interactions & threshold effects made it the clear winner. I was genuinely surprised the neural network didn't perform better , this led me to read more about why tree ensembles often dominate structured tabular data even in an era of deep learning hype

### Feature importance tells a story

From the final RF model:
1. **Baths (56.7%)** — strongest predictor, more than beds or raw area
2. **Location encoding (20.5%)** — neighborhood effects are huge
3. **Total_Rooms (8.1%)** — my engineered livability metric worked

This suggests renters prioritize *functional comfort* (bathrooms, usable space) over pure square footage & pay substantial premiums for specific locations. It makes sense for a market with many expatriate professionals who value convenience & amenities over sprawling space

### Where models break down

Error analysis showed prediction accuracy degrades significantly for luxury properties (top price decile). I tried training separate models for normal vs luxury segments, but the global Random Forest actually performed better, suggesting the single ensemble already captures segment differences effectively or perhaps that the luxury segment has genuinely higher irreducible variance due to unobserved features (views, specific building amenities, etc.)

---

## Technical Choices I Made

**Log transformation on target:** Raw rents are heavily right skewed (skewness ~5.7). Log transformation reduced this to ~0.39, stabilizing variance for the models.

**Target encoding with smoothing:** Simple one-hot encoding would create 85 sparse location features. Target encoding compresses this to one meaningful number while preserving signal. I used smoothing (alpha=10) & a minimum samples threshold to handle rare neighborhoods safely, critical since I couldn't visually verify if a location with 3 listings was genuinely cheap or just noisy.

**Dropped Rent_per_sqft:** Initially present in the data but this is target leakage—it's derived from rent itself. Removing it forced the model to learn pricing from fundamentals

**No imputation for missing Beds:** ~9.6% of listings had missing bed counts. These were almost exclusively apartments suggesting data collection gaps rather than random missingness. I removed these rows rather than imputing since a property without bedroom information can't realistically represent a valid rental unit

---

## Tools & Libraries

- **pandas** — data manipulation & feature engineering
- **scikit-learn** — preprocessing, model training, evaluation pipelines
- **matplotlib / seaborn** — exploratory visualization
- **Jupyter** — iterative development & analysis

---

## Running This

The project is organized as a single Jupyter notebook (`abu_dhabi_rent_analysis.ipynb`) that runs end-to-end. Data cleaning, EDA, feature engineering, model training & evaluation are all documented with inline commentary explaining my reasoning at each step


## Setup

```bash
pip install -r requirements.txt
jupyter notebook notebooks/abu_dhabi_rent_analysis.ipynb