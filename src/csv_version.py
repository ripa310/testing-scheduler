import math

import pandas as pd
from collections import defaultdict
from itertools import permutations
import time

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
        for col in df.columns[1:18]
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
        max_raids_per_day += 3  # Increase max raids per day by 3
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


for timeslot, group in grouped_raids:

    print(f"Timeslot: {timeslot}")
    print(group.reset_index(drop=True), "\n")


def get_names(raid):
    return list(set([row for rows in raid for row in rows]))


def get_raid(grouped_raids, day):
    return grouped_raids[["name", "members", "timeslot"]].apply(lambda x: x).loc[day]["members"].apply(lambda x: [list(d.values())[0] for d in x]).reset_index(drop=True)

def get_raid_with_class(grouped_raids, day):
    return grouped_raids[["name", "members", "timeslot"]].apply(lambda x: x).loc[day]["members"].apply(lambda x: [list(d.values()) for d in x]).reset_index(drop=True)

def get_raid_names(grouped_raids, day):
    return grouped_raids[["name", "members", "timeslot"]].apply(lambda x: x).loc[day]["name"].reset_index(drop=True)
def distance(name, raid_day):
    d = 0
    not_in_raid = 0
    for i,raid in enumerate(raid_day):
        if name in raid:
            for raid2 in raid_day[i:]:
                if name in raid2:
                    d+=not_in_raid
                    not_in_raid = 0
                else:
                    not_in_raid+=1
            return d

def total_distance2(raid_day, list_of_names, best):
    total_dist = 0
    for name in list_of_names:
        dist = distance(name, raid_day)
        total_dist += dist
        if total_dist >= best:
            return float("inf")
    return total_dist
def total_distance(raid_day):
    all_names = set(name for raid in raid_day for name in raid)
    return sum(distance(name, raid_day) for name in all_names)
def optimal_permutation(raid, list_of_names):

    min_distance = float("inf")
    optimal_order = None
    for perm in permutations(raid):
        dist = total_distance2(perm, list_of_names, min_distance)
        if dist < min_distance:
            min_distance = dist
            optimal_order = perm
    return min_distance, optimal_order


def optimal_perm_for_day(grouped_raids, day):
    raid = get_raid(grouped_raids, day)
    names = get_names(raid)
    return optimal_permutation(raid, names)


# Branch-and-bound function
def branch_and_bound(series):
    n = len(series)
    best_distance = float('inf')
    best_permutation = None
    perm = []

    def bound(partial_permutation):
        """Compute a lower bound for the partial permutation."""
        partial_distance = total_distance(partial_permutation)
        # Minimal estimate for remaining distances
        return partial_distance

    def recurse(partial_permutation, current_perm):
        nonlocal best_distance, best_permutation, perm

        # If the permutation is complete, calculate its total distance
        if len(partial_permutation) == n:
            total_dist = total_distance(partial_permutation)
            if total_dist < best_distance:
                best_distance = total_dist
                best_permutation = partial_permutation
                perm = current_perm
            return

        # Check bound for the current partial permutation
        if bound(partial_permutation) >= best_distance:
            return  # Prune this branch

        # Branch: Extend the permutation by adding each remaining element
        for i,next_element in enumerate(series):
            if next_element not in partial_permutation:
                recurse(partial_permutation + [next_element], current_perm + [i])

    # Start the recursive branching
    recurse([], [])

    return best_distance, best_permutation, perm



start = time.time()
min_distance, optimal_order, permutation = branch_and_bound(get_raid(grouped_raids, "Fri"))
print("Optimal Order:", optimal_order)
print("Minimum Distance:", min_distance)
print("Permutation:", permutation)
end = time.time()
print(end - start)
print(get_raid_with_class(grouped_raids, "Fri").reindex(permutation).reset_index(drop=True))
print(get_raid_names(grouped_raids, "Fri").reindex(permutation).reset_index(drop=True))
html_output = ""
expanded_row = []
df = pd.DataFrame(columns=["timeslot", "name", "members"])
for timeslot, _ in grouped_raids:
    min_distance, optimal_order, permutation = branch_and_bound(get_raid(grouped_raids, timeslot))
    raid_names = get_raid_names(grouped_raids, timeslot).reindex(permutation).reset_index(drop=True)
    name_and_class = get_raid_with_class(grouped_raids, timeslot).reindex(permutation).reset_index(drop=True)
    html_output += f"<h2>Timeslot: {timeslot}</h2>"
    html_output += pd.concat([raid_names, name_and_class], axis=1).to_html(index=False)
    print(df)
    df = pd.concat([df, pd.concat([pd.Series([timeslot] * len(raid_names), name="timeslot"), raid_names, name_and_class], axis=1)], ignore_index=True)
    print(pd.concat([pd.Series([timeslot] * len(raid_names), name="timeslot"), raid_names, name_and_class], axis=1))



print(df["members"])
max_length = df['members'].apply(len).max()  # Find the longest list
expanded_lists = pd.DataFrame(df['members'].tolist(), columns=[f'member{i+1}' for i in range(max_length)])

# Concatenate the expanded lists with the original DataFrame
df_expanded = pd.concat([df.drop(columns=['members']), expanded_lists], axis=1)

# Save the DataFrame to a CSV file
df_expanded.to_csv('result_schedule.csv', index=False)



with open("result_schedule.html", "w") as f:
    f.write(html_output)
