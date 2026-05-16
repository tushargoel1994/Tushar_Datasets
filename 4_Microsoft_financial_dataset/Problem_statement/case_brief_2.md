
# Microsoft Finance Sample Dataset

## Introduction

This dataset is presented by Microsoft for basic power BI analysis. The dataset contains the sales and profitability details of 6 products across 5 segments and 5 countries in 2013 to 2014. The goal of this project is to analyze the current sales data, find trends and prepare suggestions to increase the profitability of Microsoft. 

## Dataset Introduction

### Existing Data Table Format
| Column | Data Range | Description | Notes/ Assumption |
| --- | --- | --- | --- |
| Segment | Channel Partners, Midmarket, Government, Enterprise, Small Business | Different customers company sell its products to | Primary differentiator as same product has siginificant different pricing & cost in different segments |
| Country | Canada, France, Germany, Mexico, United States | Sales Geographies | Unlimited supply of each product available at country specific cost |
| Product | Amarilla, Caraterra, Montana, Paseo, Velo, VTT | Substitute products microsoft selling into each market | one product can substitute other |
| Discount Band | None, Low, Medium High | Discounts offered by Microsoft | **Need analysis if discount bands are product/ or geography specific** |
| Units Sold | 200 - 5000 | Units sold per month per product, segment and geography | **.5 product can be sold, tricky** |
| Manufacturing Price | $3 - $260 | cost to manufacture one unit | **not relevant as cost < manufacturing for few semgents that eliminates the requirement of this column** |
| Sale Price | $7 - $350 | Undiscounted listed price | Apart for govt, remained same for all products per segment | 
| Gross Sales | upto $1.22 million | sale price * units sold (Discounts not considered) | sales per product, segment, geography, month |
| Discounts | upto 160K | Total discount provided per product, segment, geography, month | In direct relations with sales |
| Sales | upto $1.2 million | Gross sales - Discounts| Actual sales |
| COGS | upto $960k | Cost of Goods sold per product, segment, geography, month | **Need to explore if in direct relation with manufacturing price |
| Profit | upto $270K | Actual Sales - COGS | | 
| Date| Sep 2013 - Dec 2014 | Observation Time Range | Already divided Month Number, Month Name and Year |

### new Columns Introduced
First task is to bring add additional columns that make data analysis easy and simplified

| Column Name | Source Column(s) | Description |
| --- | --- | --- |
| Discount Percentage | Discounts, Gross Sales | Discount offered in percentage | 
| Discounted Price Per Unit | Sales, Units Sold | Actual price units are sold in particular group |
| Cost Per Unit | COGS, Units Sold | Cost per unit |
| Profit Per Unit | Profits, Units Sold | Profit earned selling each unit |
| Profit Margin Percentage | Profit Per Unit, Discounted Price Per Unit | Margins Earned |

## A. Exploratory Data Analysis

Use Pivot table to create initial Analysis

### Table A(1) - Sales Price of Product across Segments (Filtered By Countries)
The observations shows that sale price remain same across products and countries for same segment but vary substantially across segments. 

### Table A(2) - Cost Price of Product across Segment (Filterd By Countries)
The cost price remain same across products and countries for same segment but vary substantially across segments. 

### Table A(3) - Discount Price of Products across Segments (All Countries)
This shows discounted price (or discount percentage) vary across products and countries as well. Prices across segments vary substantially and are not comparable.

The observations led to conclusion that Listed sales price and cost remain same across products and countries (Except for government segment) but vary across segment. Also, separate discounts are given across products and countries as well. This lead to conclusion that 2 segments are not comparable and each segment should be analyzed separately.


## B. Channel Partner Segment Analysis
In this section, for channel partners we try to conclude how price and profit varies across products and countries. We also look at the trend in price across time and see if any particular trend is observable.

### Table B(1): Average Discount Offered (%) across Products and Countries (Segment: Channel Partner)
Discounts offered vary by country and product with average value between 6-8% for each combination in case of Channel Partners

### Table B(2): Average Profit Margins Earned across Products and Countries (Segment: Channel Partner)
Average Profit Margin remained between 70-75% but vary across each combination due to discounts. 

### Tabl3 B(3): Discount Percentage Provided across Combination of Product, Countries and Time Period (Segment: Channel Partner)
Discounts vary across each entry. However, if a discount is offered in one country, there is a high probability it will be offered in other countries as well (not everytime), however the degree of discounts vary substantially. For our analysis, we can consider that discounts are independet of each other and Microsoft can provide selective discounts in different product/ country combination at will.



## Goal
The goal is to figure out the following:

### For a particular segment say Channel Partner
- See Change is consumption pattern of different products across all geographies over time
- Look at product cannabilization in different geographies, is one product taking over others in this particular segment on distribution remains the same?
- Are profits declining on month on month basis. In which countries the profits have declined the most or which countries are more profitable.
- What is the impact of discount on profits, are decresing prices incresing profits?
- Is unit sales 
