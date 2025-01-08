import numpy as np
import math
from datetime import datetime, timedelta
import itertools
import pulp
import pandas as pd
from collections import defaultdict
from itertools import permutations, combinations
from pulp import LpProblem, LpVariable, LpMinimize, lpSum, value



file_path = "Lost Ark - Raid Comps.csv"
df = pd.read_csv(file_path)
awei_path = "Lost Ark - Awei.csv"

awei_df = pd.read_csv(awei_path, header=None)

awei_df = awei_df.set_index(0)

member_unavailability = awei_df.apply(
    lambda row: {day for day in row if pd.notna(day)}, axis=1
).to_dict()

available_path = "Lost Ark - Awei 2.csv"
test_available_df = pd.read_csv(available_path)

def create_member_availability(df):
    member_availability = {}
    for _, row in df.iterrows():
        name = row["Name"]
        member_availability[name] = {}
        for day in df.columns[1:]:  # Loop through the days of the week
            start_hour, end_hour = row[day].split("-")
            # Generate availability as a single interval
            member_availability[name][day] = [(start_hour, end_hour)]
    return member_availability

member_availability2 = create_member_availability(test_available_df)

# Print the result
print(member_availability2)

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



days_of_week = { "Wed", "Thu", "Fri", "Sat", "Sun", "Mon", "Tue"}


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


def get_len_of_raid_day(raids):
    res = len(raids)
    raid_names = [raid['name'] for raid in raids]
    behemoth_counter = raid_names.count("Behemoth")
    res =- behemoth_counter
    return res // 1


def schedule_raids(raids, days_of_week, max_raids_per_day=8, max_backtrack_attempts=2):
    scheduled_raids = defaultdict(list)
    failed_raids = []
    unscheduled_raids = []
    custom_order = ["Wed", "Thu", "Fri", "Sat", "Sun", "Mon", "Tue"]
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
                available_days  &= member_availability.get(member, set())

            available_days = sorted(available_days, key= custom_order.index)
            # Try to schedule the raid on one of the valid days
            scheduled = False
            for day in sorted(available_days, key= custom_order.index):
                if get_len_of_raid_day(scheduled_raids[day]) < max_raids_per_day:
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
            available_days = sorted(available_days, key= custom_order.index)
            # Try to find an available day and move an already scheduled raid if necessary
            for day in sorted(available_days, key= custom_order.index):
                if get_len_of_raid_day(scheduled_raids[day]) < max_raids_per_day:
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
        max_raids_per_day += 2  # Increase max raids per day by 3
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
                    available_days = sorted(available_days, key= custom_order.index)
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



min_distance, optimal_order, permutation = branch_and_bound(get_raid(grouped_raids, "Fri"))

html_output = ""
expanded_row = []
df = pd.DataFrame(columns=["timeslot", "name", "members"])



for timeslot, group in grouped_raids:
    #min_distance, optimal_order, permutation = branch_and_bound(get_raid(grouped_raids, timeslot))
    #raid_names = get_raid_names(grouped_raids, timeslot).reindex(permutation).reset_index(drop=True)
    #name_and_class = get_raid_with_class(grouped_raids, timeslot).reindex(permutation).reset_index(drop=True)
    #html_output += f"<h2>Timeslot: {timeslot}</h2>"
    #html_output += pd.concat([raid_names, name_and_class], axis=1).to_html(index=False)
    #df = pd.concat([df, pd.concat([pd.Series([timeslot] * len(raid_names), name="timeslot"), raid_names, name_and_class], axis=1)], ignore_index=True)
    print("exit")
flattened_data = []

# Loop through the grouped raids
for timeslot, group in grouped_raids:
    # Iterate over each row in the DataFrame
    for _, row in group.iterrows():
        # Extract raid name and members
        raid_name = row['name']
        members = row['members']  # This should be a list of dictionaries

        # Check if members is valid
        if isinstance(members, list):
            # Concatenate member names and classes into a single string
            member_names = [member['name'] for member in members]
            member_classes = [member['class'] for member in members]

            # Join them into a single string (comma-separated)
            member_names_str = ", ".join(member_names)
            member_classes_str = ", ".join(member_classes)

            flattened_data.append({
                'Timeslot': timeslot,
                'Raid': raid_name,
                'Member Names': member_names_str,
                'Member Classes': member_classes_str,
            })
        else:
            print(f"Unexpected data format in members: {members}")

# Convert to DataFrame
df = pd.DataFrame(flattened_data)

# Save to CSV and HTML
csv_file = "grouped_raids.csv"
df.to_csv(csv_file, index=False)
print(f"CSV file saved: {csv_file}")

html_file = "grouped_raids.html"
df.to_html(html_file, index=False, border=1)
print(f"HTML file saved: {html_file}")


def separate_name_class(members):
    names = []
    classes = []
    for member in members:
        names.append(member[0])  # Name is the first element of each list
        classes.append(member[1])  # Class is the second element of each list
    return names, classes

# Apply the function to each row in the 'members' column


with open("result_schedule.html", "w") as f:
    f.write(html_output)
    


def calculate_daily_availability(member_availability):
    """
    Calculate the total daily availability for all members.
    :param member_availability: Dictionary with member availability.
    :return: Dictionary with total availability in hours for each day.
    """
    daily_availability = defaultdict(int)

    for member, days in member_availability.items():
        for day, time_slots in days.items():
            for start, end in time_slots:
                # Handle edge case where start and end are '00:00' (indicating no availability)
                if start == "00:00" and end == "00:00":
                    continue

                start_hour, start_minute = map(int, start.split(':'))
                end_hour, end_minute = map(int, end.split(':'))

                if end == "00:00":  # Special case for midnight
                    end_hour = 24

                start_time = start_hour + start_minute / 60
                end_time = end_hour + end_minute / 60

                daily_availability[day] += end_time - start_time

    return daily_availability

def schedule_raids2(raids, member_availability, raid_duration=0.5, max_raids_per_day=8):
    """
    Schedule raids based on member availability.
    :param raids: List of raid dictionaries.
    :param member_availability: Member availability data.
    :param raid_duration: Duration of each raid in hours (default 0.5 hours).
    :param max_raids_per_day: Maximum number of raids allowed per day.
    :return: Tuple of scheduled raids and unscheduled raids.
    """
    daily_availability = calculate_daily_availability(member_availability)

    # Filter out days with insufficient total availability
    valid_days = {day for day, hours in daily_availability.items() if hours >= raid_duration * max_raids_per_day}

    scheduled_raids = defaultdict(list)
    unscheduled_raids = []

    for raid in raids:
        scheduled = False
        for day in valid_days:
            if len(scheduled_raids[day]) < max_raids_per_day:
                scheduled_raids[day].append(raid)
                raid['timeslot'] = day
                scheduled = True
                break

        if not scheduled:
            unscheduled_raids.append(raid)

    return scheduled_raids, unscheduled_raids




print(unscheduled_raids)

# Example series abcde ab abcfg ahij abkl
series = get_raid(grouped_raids, "Thu")
min_distance, optimal_order, permutation = branch_and_bound(get_raid(grouped_raids, "Thu"))
# Define your distance metric
def calculate_distance(list1, list2):
    set1, set2 = set(list1), set(list2)
    return len(set(list1).symmetric_difference(set(list2)))
# Convert series to list of nodes
# Create a distance matrix for the lists
n = len(series)
distance_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        if i != j:
            distance_matrix[i][j] = calculate_distance(series[i], series[j])

# ILP formulation to minimize the total distance
prob = pulp.LpProblem("MinimumHamiltonianPath", pulp.LpMinimize)

# Decision variables: x[i][j] is 1 if the path goes from list i to list j
x = pulp.LpVariable.dicts("x", (range(n), range(n)), cat='Binary')

# Objective: Minimize the total distance
prob += pulp.lpSum(distance_matrix[i][j] * x[i][j] for i in range(n) for j in range(n) if i != j)

# Constraints:
# 1. Each list must be visited exactly once (entering)
for i in range(n):
    prob += pulp.lpSum(x[j][i] for j in range(n) if j != i) == 1, f"Enter_list_{i}"

# 2. Each list must be visited exactly once (leaving)
for i in range(n):
    prob += pulp.lpSum(x[i][j] for j in range(n) if i != j) == 1, f"Leave_list_{i}"

# Solve the ILP
prob.solve()

# Extract the optimal path
# Check if the solution is optimal
if pulp.LpStatus[prob.status] != 'Optimal':
    print("No optimal solution found.")
else:
    # Extract the optimal path from the solution
    path = []
    for i in range(n):
        for j in range(n):
            if pulp.value(x[i][j]) == 1:
                path.append((i, j))

    # Reconstruct the optimal sequence of lists
    visited = set()
    sequence = []
    current_node = 0  # Start from the first list (could be any, as we are solving for the full path)

    while len(visited) < n:
        sequence.append(series[current_node])
        visited.add(current_node)
        # Find the next node to visit
        next_node = None
        for j in range(n):
            if pulp.value(x[current_node][j]) == 1 and j not in visited:
                next_node = j
                break
        if next_node is None:
            break
        current_node = next_node

    # Print the optimal permutation of the lists
    print("Optimal sequence of lists:")
    for seq in sequence:
        print(seq)


d1 = 0
d2 = 0
for i in range(len(optimal_order)-1):

    d2 += calculate_distance(optimal_order[i], optimal_order[i+1])



print(d2)
print(optimal_order)

for i,a in enumerate(prob.variables()):
    if i % 8 == 0:
        print()
    print(a.varValue)





