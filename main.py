# Stack implementation in python


# Creating a stack
def create_stack():
    stack = []
    return stack


# Creating an empty stack
def check_empty(stack):
    return len(stack) == 0
# if it returns true, it means that it is empty.

# Adding items into the stack
def push(stack, item):
    stack.append(item)
    print("pushed item: " + item)


# Removing an element from the stack
def pop(stack):
    if (check_empty(stack)):
        return "stack is empty"

    return stack.pop()

def peek(stack):
  if (check_empty(stack)):
    return "stack is empty"
  return stack[-1]

#stack = create_stack()
#push(stack, str(1))
#push(stack, str(2))
#push(stack, str(3))
#push(stack, str(4))
#print ("peeking" + peek(stack))
#print("popped item: " + pop(stack))
#print("stack after popping an element: " + str(stack))


unsorted = [3, 1, 2, 4, 5, 6, 8, 9]

def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):  # Last i elements are already in place
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]  # Swap if the element found is greater than the next element

def quicksort(arr, low, high):
    if low < high:
        # Partition the array and get the pivot index
        pi = partition(arr, low, high)

        # Recursively sort the elements before and after partition
        quicksort(arr, low, pi - 1)
        quicksort(arr, pi + 1, high)

def partition(arr, low, high):
    # Choose the rightmost element as pivot
    pivot = arr[high]
    i = low - 1  # Index of smaller element

    for j in range(low, high):
        # If current element is smaller than or equal to pivot
        if arr[j] <= pivot:
            i += 1  # Increment index of smaller element
            arr[i], arr[j] = arr[j], arr[i]  # Swap

    # Swap the pivot element with the element at i+1
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

# Example usage:
#arr = [3, 6, 8, 10, 1, 2, 1]
#print("Original array:", arr)
#quicksort(arr, 0, len(arr) - 1)
#print("Sorted array:", arr)


# Example usage:
quicksort(unsorted, 0, len(unsorted) - 1)
print("Sorted array is:", unsorted)
  
