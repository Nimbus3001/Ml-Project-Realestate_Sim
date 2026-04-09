def classify_growth(building_density, road_density):
    score = 0.6 * building_density + 0.4 * road_density

    print("Score:", score)

    # NEW improved logic
    if score < 0.02:
        return "Low"
    elif score < 0.15:
        return "High"
    else:
        return "Medium"