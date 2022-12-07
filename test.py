import numpy as np
import subprocess


test_cases = 10
test_size = 5

print("""
        CSR matrice test script\n
        Since floating point precision differs between base program and the test script, these tests ensure error between them is bellow some treshold.
""")
for i in range(test_cases):
    print("Test case: " ,i)

    #Generate random array
    random_arr = np.random.randint(0,2, (test_size, test_size))
    random_arr[0] = np.ones(test_size)
    x_sum =  np.sum(random_arr, axis= 1)
    for i in range(test_size):
        if x_sum[i]  == 0:
            random_arr[i][0] = 1
    y_sum =  np.sum(random_arr, axis= 0)
    for i in range(test_size):
        if y_sum[i]  == 0:
            random_arr[0][i] = 1
    random_arr = np.array(random_arr, dtype= float)
    random_arr /= np.sum(random_arr, axis= 0)
    random_arr = np.nan_to_num(random_arr)

    f = open("test.test", "w")
    for i in range(test_size):
        for k in range(test_size):
            if random_arr[i][k] > 0:
                # print(i,k)
                f.write(f"{k} {i}\n")
    f.close()
    #Convert to CSR matrice
    row_begin = [0]
    values = []
    col_indices = []
    for i in range(test_size):
        col_end = 0
        for k in range(test_size):
            val = random_arr[i][k]
            if val != 0:
                col_end += 1
                values.append(val)
                col_indices.append(k)
        row_begin.append(row_begin[-1]  + col_end)

    #prepare the test file for the program

    #execute the program
    subprocess.run(["g++-12 -c eee1.cpp -fopenmp -o test.o -DPRINT_CSR; g++-12 test.o -o test -fopenmp -lpthread; ./test test.test"], shell=True)


    f = open("csr.txt", "r")
    lines = f.readlines()
    lines = [line.split(' ') for line in lines]
    row_begin_result = [int(val) for val in lines[0][:-1]]
    values_result = [float(val) for val in lines[1][:-1]]
    col_indices_result = [int(val) for val in lines[2][:-1]]
    f.close()
    # print(random_arr)
    # print("result is ", row_begin_result)
    # print("correct answer is ",row_begin)
    x = np.sum(np.abs(np.subtract(np.array(values_result, dtype=float), np.array(values, dtype=float))))
    
    assert(np.array_equal(np.array(row_begin_result), np.array(row_begin)))
    assert(x < 1e-3)
    assert(np.array_equal(np.array(col_indices_result), np.array(col_indices)))
    print("CSR matrice constructed without problems")
    print("Average error between matrices:" , x/len(values))


    c = np.ones(test_size)
    r = np.ones(test_size)
    r_next = np.zeros(test_size)
    alpha = 0.2
    dif = 1
    count = 0
    while( dif > 1e-6):
        r_next = (random_arr@r * alpha) + (1 - alpha)
        dif = np.sum(np.abs(r - r_next))
        # print(r_next)
        count += 1 
        r = r_next

    f = open("top5.txt", "r")
    lines = f.readlines()
    lines = [line.split(':') for line in lines]
    
    top_5_c = []
    for val in lines:
        top_5_c.append(float(val[1]))
       
    
    top_5_c.sort()
    top_5 = np.sort(r)[:5]
    print("Total error in rankings: ", np.sum(np.abs(np.array(top_5_c)- top_5)) )

    assert(np.sum(np.abs(np.array(top_5_c)- top_5)) < 1e-3, "Total error in rankings exceed 1e-3")
    
    



    