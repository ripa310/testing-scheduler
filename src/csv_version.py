import pandas as pd
from collections import defaultdict

file_path = "Lost Ark - secret resake tech.csv"
df = pd.read_csv(file_path)
awei_path = "Lost Ark - more resake awei tech.csv"

awei_df = pd.read_csv(awei_path, header=None)

awei_df = awei_df.set_index(0)

member_unavailability = awei_df.apply(
    lambda row: {day for day in row if pd.notna(day)}, axis=1
).to_dict()





member_unavailability = {
    member: set(days) if isinstance(days, set) else set()
    for member, days in member_unavailability.items()
}

raids = []

raids = []
for i in range(0, len(df), 2):
    raid_row = df.iloc[i+1]
    class_row = df.iloc[i]

    # Extract raid name and members
    raid_name = raid_row["Raidname"]
    raid_members = [
        {
            "name": raid_row[col].strip(),
            "class": class_row[col].strip(),
        }
        for col in df.columns[1:9]
        if pd.notna(raid_row[col]) and pd.notna(class_row[col])
    ]

    # Add raid to the list
    raids.append({"name": raid_name, "members": raid_members, "timeslot": None})



days_of_week = {"Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"}


member_availability = {
    member: days_of_week - unavailable
    for member, unavailable in member_unavailability.items()
}

def raid_priority(raid, member_availability):
    raid_members = [member["name"] for member in raid["members"]]
    return min(len(member_availability.get(member, set())) for member in raid_members)

# Sort raids by priority
raids.sort(key=lambda raid: raid_priority(raid, member_availability))

for raid in raids:
    print(raid)

unavailability = {name: set(days) for name, days in member_unavailability.items()}

scheduled_raids = defaultdict(list)
unscheduled_raids = []


def schedule_raids(raids, days_of_week, max_raids_per_day=8, max_backtrack_attempts=2):
    scheduled_raids = defaultdict(list)
    failed_raids = []
    unscheduled_raids = []

    # Function to check if a raid can be scheduled on a specific day
    def can_schedule(raid, day):
        raid_members = [member["name"] for member in raid["members"]]
        return all(day in member_availability.get(member, set()) for member in raid_members)

    # Function to try to schedule all raids
    def try_schedule():
        for raid in raids:
            raid_members = [member["name"] for member in raid["members"]]
            available_days = set(days_of_week)

            # Find valid days for each raid by checking member availability
            for member in raid_members:
                available_days &= member_availability.get(member, set())

            # Try to schedule the raid on one of the valid days
            scheduled = False
            for day in sorted(available_days):
                if len(scheduled_raids[day]) < max_raids_per_day:
                    scheduled_raids[day].append(raid)
                    raid["timeslot"] = day
                    scheduled = True
                    break

            if not scheduled:
                failed_raids.append(raid)

        return not failed_raids  # If no failed raids, all are scheduled

    # Try greedy scheduling
    success = try_schedule()

    # Backtracking attempts
    backtrack_attempts = 0
    while not success and backtrack_attempts < max_backtrack_attempts:
        print(f"Backtracking attempt {backtrack_attempts + 1}...")

        # Backtrack by attempting to move already scheduled raids to make space
        for raid in failed_raids[:]:
            raid_members = [member["name"] for member in raid["members"]]
            available_days = set(days_of_week)

            for member in raid_members:
                available_days &= member_availability.get(member, set())

            # Try to find an available day and move an already scheduled raid if necessary
            for day in sorted(available_days):
                if len(scheduled_raids[day]) < max_raids_per_day:
                    scheduled_raids[day].append(raid)
                    raid["timeslot"] = day
                    failed_raids.remove(raid)
                    break

        # Check if there are any failed raids remaining after backtracking
        success = not failed_raids
        backtrack_attempts += 1

    # If there are still unscheduled raids after two backtracking attempts, increase max raids per day
    if not success and backtrack_attempts >= max_backtrack_attempts:
        print(f"Could not schedule all raids after {max_backtrack_attempts} backtracking attempts. Increasing max raids per day.")
        max_raids_per_day += 4  # Increase max raids per day by 2
        # Retry scheduling with the updated max raids per day
        scheduled_raids = defaultdict(list)
        failed_raids = []

        # Try scheduling again with increased max raids per day
        success = try_schedule()

        # If there are still failed raids after the second attempt, we try backtracking again
        if not success:
            backtrack_attempts = 0
            while not success and backtrack_attempts < max_backtrack_attempts:
                print(f"Backtracking attempt {backtrack_attempts + 1}...")

                # Backtrack again to try and schedule the failed raids
                for raid in failed_raids[:]:
                    raid_members = [member["name"] for member in raid["members"]]
                    available_days = set(days_of_week)

                    for member in raid_members:
                        available_days &= member_availability.get(member, set())

                    # Try to find an available day and move an already scheduled raid if necessary
                    for day in sorted(available_days):
                        if len(scheduled_raids[day]) < max_raids_per_day:
                            scheduled_raids[day].append(raid)
                            raid["timeslot"] = day
                            failed_raids.remove(raid)
                            break

                # Check if there are any failed raids remaining after the second backtracking attempt
                success = not failed_raids
                backtrack_attempts += 1

    # After the backtracking attempts, store unscheduled raids
    if failed_raids:
        unscheduled_raids = failed_raids

    return scheduled_raids, unscheduled_raids, success


scheduled_raids, unscheduled_raids, success = schedule_raids(raids, days_of_week)

result_df = pd.DataFrame(raids)


grouped_raids = result_df.groupby('timeslot')
html_output = ""
for timeslot, group in grouped_raids:
    html_output += f"<h2>Timeslot: {timeslot}</h2>"
    html_output += group.to_html(index=False)

with open("grouped_raids.html", "w") as f:
    f.write(html_output)


test_df = pd.DataFrame(grouped_raids)

for timeslot, group in grouped_raids:
    print(f"Timeslot: {timeslot}")
    print(group, "\n")

expanded_rows = []

# Loop over each timeslot group
for timeslot, group in grouped_raids:
    # For each group, prepare the columns for the raid's members
    for _, row in group.iterrows():
        expanded_row = {'timeslot': timeslot, 'name': row['name']}

        # Add each member's name and class (combined with ':') to its own column
        for i, member in enumerate(row['members']):
            expanded_row[f"member_{i+1}"] = f"{member['name']}:{member['class']}"

        # Append the expanded row to the list
        expanded_rows.append(expanded_row)

# Convert the list of expanded rows into a new DataFrame
expanded_df = pd.DataFrame(expanded_rows)

# Save the expanded DataFrame to a CSV file
expanded_csv_filename = 'grouped_raids_expanded_combined_columns.csv'
expanded_df.to_csv(expanded_csv_filename, index=False)
