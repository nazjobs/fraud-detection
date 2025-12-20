import pandas as pd
import ipaddress


def transform_ip_to_int(ip):
    """
    Transforms an IP address string to its integer representation.
    Returns 0 if conversion fails.
    """
    try:
        if isinstance(ip, str):
            # Using ip_address to handle standard IPv4 notation
            return int(ipaddress.ip_address(ip))
        return int(ip)
    except:
        return 0


def merge_fraud_data_with_geolocation(fraud_df, ip_country_df):
    """
    Merges fraud data with IP country data using merge_asof.
    Assumes fraud_df has 'ip_address' and ip_country_df has lower/upper bounds.
    """
    # 1. Convert IP to int
    print("Converting IPs to integers...")
    fraud_df["ip_address_int"] = fraud_df["ip_address"].apply(transform_ip_to_int)

    # 2. Ensure country data types are int
    ip_country_df["lower_bound_ip_address"] = ip_country_df[
        "lower_bound_ip_address"
    ].astype(int)
    ip_country_df["upper_bound_ip_address"] = ip_country_df[
        "upper_bound_ip_address"
    ].astype(int)

    # 3. Sort for merge_asof
    fraud_df = fraud_df.sort_values("ip_address_int")
    ip_country_df = ip_country_df.sort_values("lower_bound_ip_address")

    # 4. Merge
    print("Merging data (this may take a moment)...")
    merged_df = pd.merge_asof(
        fraud_df,
        ip_country_df,
        left_on="ip_address_int",
        right_on="lower_bound_ip_address",
        direction="backward",
    )

    # 5. Filter valid ranges
    # The IP must be less than the upper bound of the matched country range
    merged_df = merged_df[
        merged_df["ip_address_int"] <= merged_df["upper_bound_ip_address"]
    ]

    return merged_df
