METADATAS = [
    {
        "name": "Wireless Headphones",
        "code": "WH001",
        "price": 149.99,
        "is_available": True,
        "release_date": "2023-10-26",
        "tags": ["audio", "wireless", "electronics"],
        "dimensions": [18.5, 7.2, 21.0],
        "inventory_location": [101, 102],
        "available_quantity": 50,
    },
    {
        "name": "Ergonomic Office Chair",
        "code": "EC002",
        "price": 299.00,
        "is_available": True,
        "release_date": "2023-08-15",
        "tags": ["furniture", "office", "ergonomic"],
        "dimensions": [65.0, 60.0, 110.0],
        "inventory_location": [201],
        "available_quantity": 10,
    },
    {
        "name": "Stainless Steel Water Bottle",
        "code": "WB003",
        "price": 25.50,
        "is_available": False,
        "release_date": "2024-01-05",
        "tags": ["hydration", "eco-friendly", "kitchen"],
        "dimensions": [7.5, 7.5, 25.0],
        "available_quantity": 0,
    },
    {
        "name": "Smart Fitness Tracker",
        "code": "FT004",
        "price": 79.95,
        "is_available": True,
        "release_date": "2023-11-12",
        "tags": ["fitness", "wearable", "technology"],
        "dimensions": [2.0, 1.0, 25.0],
        "inventory_location": [401],
        "available_quantity": 100,
    },
]

FILTERING_TEST_CASES = [
    # These tests only involve equality checks
    (
        {"code": "FT004"},
        ["FT004"],
    ),
    # String field
    (
        # check name
        {"name": "Smart Fitness Tracker"},
        ["FT004"],
    ),
    # Boolean fields
    (
        {"is_available": True},
        ["WH001", "FT004", "EC002"],
    ),
    # And semantics for top level filtering
    (
        {"code": "WH001", "is_available": True},
        ["WH001"],
    ),
    # These involve equality checks and other operators
    # like $ne, $gt, $gte, $lt, $lte
    (
        {"available_quantity": {"$eq": 10}},
        ["EC002"],
    ),
    (
        {"available_quantity": {"$ne": 0}},
        ["WH001", "FT004", "EC002"],
    ),
    (
        {"available_quantity": {"$gt": 60}},
        ["FT004"],
    ),
    (
        {"available_quantity": {"$gte": 50}},
        ["WH001", "FT004"],
    ),
    (
        {"available_quantity": {"$lt": 5}},
        ["WB003"],
    ),
    (
        {"available_quantity": {"$lte": 10}},
        ["WB003", "EC002"],
    ),
    # Repeat all the same tests with name (string column)
    (
        {"code": {"$eq": "WH001"}},
        ["WH001"],
    ),
    (
        {"code": {"$ne": "WB003"}},
        ["WH001", "FT004", "EC002"],
    ),
    # And also gt, gte, lt, lte relying on lexicographical ordering
    (
        {"name": {"$gt": "Wireless Headphones"}},
        [],
    ),
    (
        {"name": {"$gte": "Wireless Headphones"}},
        ["WH001"],
    ),
    (
        {"name": {"$lt": "Smart Fitness Tracker"}},
        ["EC002"],
    ),
    (
        {"name": {"$lte": "Smart Fitness Tracker"}},
        ["FT004", "EC002"],
    ),
    (
        {"is_available": {"$eq": True}},
        ["WH001", "FT004", "EC002"],
    ),
    (
        {"is_available": {"$ne": True}},
        ["WB003"],
    ),
    # Test float column.
    (
        {"price": {"$gt": 200.0}},
        ["EC002"],
    ),
    (
        {"price": {"$gte": 149.99}},
        ["WH001", "EC002"],
    ),
    (
        {"price": {"$lt": 50.0}},
        ["WB003"],
    ),
    (
        {"price": {"$lte": 79.95}},
        ["FT004", "WB003"],
    ),
    # These involve usage of AND, OR and NOT operators
    (
        {"$or": [{"code": "WH001"}, {"code": "EC002"}]},
        ["WH001", "EC002"],
    ),
    (
        {"$or": [{"code": "WH001"}, {"available_quantity": 10}]},
        ["WH001", "EC002"],
    ),
    (
        {"$and": [{"code": "WH001"}, {"code": "EC002"}]},
        [],
    ),
    # Test for $not operator
    (
        {"$not": {"code": "WB003"}},
        ["WH001", "FT004", "EC002"],
    ),
    (
        {"$not": [{"code": "WB003"}]},
        ["WH001", "FT004", "EC002"],
    ),
    (
        {"$not": {"available_quantity": 0}},
        ["WH001", "FT004", "EC002"],
    ),
    (
        {"$not": [{"available_quantity": 0}]},
        ["WH001", "FT004", "EC002"],
    ),
    (
        {"$not": {"is_available": True}},
        ["WB003"],
    ),
    (
        {"$not": [{"is_available": True}]},
        ["WB003"],
    ),
    (
        {"$not": {"price": {"$gt": 150.0}}},
        ["WH001", "FT004", "WB003"],
    ),
    (
        {"$not": [{"price": {"$gt": 150.0}}]},
        ["WH001", "FT004", "WB003"],
    ),
    # These involve special operators like $in, $nin, $between
    # Test between
    (
        {"available_quantity": {"$between": (40, 60)}},
        ["WH001"],
    ),
    # Test in
    (
        {"name": {"$in": ["Smart Fitness Tracker", "Stainless Steel Water Bottle"]}},
        ["FT004", "WB003"],
    ),
    # With numeric fields
    (
        {"available_quantity": {"$in": [0, 10]}},
        ["WB003", "EC002"],
    ),
    # Test nin
    (
        {"name": {"$nin": ["Smart Fitness Tracker", "Stainless Steel Water Bottle"]}},
        ["WH001", "EC002"],
    ),
    ## with numeric fields
    (
        {"available_quantity": {"$nin": [50, 0, 10]}},
        ["FT004"],
    ),
    # These involve special operators like $like, $ilike that
    # may be specified to certain databases.
    (
        {"name": {"$like": "Wireless%"}},
        ["WH001"],
    ),
    (
        {"name": {"$like": "%less%"}},  # adam and jane
        ["WH001", "WB003"],
    ),
    # These involve the special operator $exists
    (
        {"tags": {"$exists": False}},
        [],
    ),
    (
        {"inventory_location": {"$exists": False}},
        ["WB003"],
    ),
]

NEGATIVE_TEST_CASES = [
    {"$nor": [{"code": "WH001"}, {"code": "EC002"}]},
    {"$and": {"is_available": True}},
    {"is_available": {"$and": True}},
    {"is_available": {"name": "{Wireless Headphones", "code": "EC002"}},
    {"my column": {"$and": True}},
    {"is_available": {"code": "WH001"}},
    {"$and": {}},
    {"$and": []},
    {"$not": True},
]
