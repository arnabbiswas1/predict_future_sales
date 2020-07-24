## Observations
- Shops
    - There are 60 unique shops

- Items 
    - There are 84 unique item categories
    - There are 22170 unique items
    - item_id is unique irrespective of it's combination with item_category_id
    
- training data
    - There are no missing values
    - [TODO] There are 56 duplicate rows based on date, shop and item_id.
    - Date ranges from Jan'2013 to Oct' 2015
    - There are 21807 unique items sold across 60 shops
    - Across shops all the items are not sold on all the days
    - For individual shops as well, all the items are not sold on all the days
    - Above two observations, indicate that prediction will be dependent on item_categories (Similar items will be sold in similar quantities)

- test date
    - Need to predict for 42 shops and 5100 items for each shop