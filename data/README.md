# Data

The pipeline reads the nine tables of the *Brazilian E-Commerce Public Dataset by Olist*, published by Olist on Kaggle
(<https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce>) under the CC BY-NC-SA 4.0 licence. The data are
not redistributed in this repository.

1. Download the data set from Kaggle and unzip it.
2. Copy the nine CSV files into `data/raw/`:

```
olist_customers_dataset.csv            olist_order_reviews_dataset.csv
olist_geolocation_dataset.csv          olist_orders_dataset.csv
olist_order_items_dataset.csv          olist_products_dataset.csv
olist_order_payments_dataset.csv       olist_sellers_dataset.csv
product_category_name_translation.csv
```

The geolocation table may be stored compressed as `olist_geolocation_dataset.csv.gz`. Before any analysis the pipeline
checks the SHA-256 of every table (for a `.gz` file, of its decompressed content) against
`src/leakaware/data/raw_sha256.json` and stops if a file differs from the release used in the article.
