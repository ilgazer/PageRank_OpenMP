#include <cstdio>
#include <omp.h>
#include <iostream>
#include <ostream>
#include <fstream>
#include <unordered_map>
#include <vector>
#include <chrono>

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

void timed_portion(int row_begin_size, int* row_begin, int* col_indices, double* values);

template <typename S> std::ostream& operator<<(std::ostream& os,
                    const std::vector<S>& vector)
{
    // Printing all the elements
    // using <<
    os << "[";
    for (S element : vector) {
        os << element << ", ";
    }
    os << "]";
    return os;
}

class Site
{
public:
    std::string name;
    int id;
    int num_outgoing_edges;
    std::vector<int> incoming_edges;

    friend std::ostream& operator<<(std::ostream& os, Site const & s) {
        return os << "Site{name="<<s.name<<" id="<<s.id<<", incoming_edges=" << s.incoming_edges << ", num_outgoing_edges="<<s.num_outgoing_edges << "}";
    }
};

static std::unordered_map<std::string, Site *> sites_by_ref;
static std::vector<Site *> sites_by_id;

Site* get_site(const std::string &name)
{
    Site *site;
    if (sites_by_ref.find(name) == sites_by_ref.end())
    {
        site = new Site{name, (int)sites_by_id.size(), 0};
        sites_by_id.push_back(site);
        sites_by_ref.insert({name, site});
    }
    else
    {
        site = sites_by_ref[name];
    }
    return site;
}

int main()
{
    std::ifstream txt("graph.txt");

    std::string first_token , second_token;
    while (txt >> first_token)
    {
        txt >> second_token;

        Site* from = get_site(first_token);
        Site* to = get_site(second_token);

        from->num_outgoing_edges++;
        to->incoming_edges.push_back(from->id);
    }


    /*for (Site* element : sites_by_id) {
        std::cout << *element  << std::endl;
    }*/

    size_t row_begin_size = sites_by_id.size() + 1;
    int* row_begin = new int[row_begin_size];

    size_t next_row_begin = 0;
    row_begin[0] = 0;
    for(size_t i = 1; i < row_begin_size; i++){
        next_row_begin += sites_by_id[i - 1]->incoming_edges.size();
        row_begin[i] = next_row_begin;
    }

    double* values = new double[next_row_begin];
    int* col_indices = new int[next_row_begin];

    for(int i = 0; i < row_begin_size - 1; i++){
        int this_row_begin = row_begin[i];
        Site* site = sites_by_id[i];
        for (size_t j = 0; j < site->incoming_edges.size(); j++)
        {
            int to = site->incoming_edges[j];
            col_indices[this_row_begin+j] = to;
            values[this_row_begin+j] = 1.0/sites_by_id[to]->num_outgoing_edges;
        }
    }

#ifdef PRINT_CSR
    print_csr(row_begin_size, next_row_begin, row_begin, col_indices, values);
#endif

    
}

void timed_portion(int row_begin_size, int* row_begin, int* col_indices, double* values)
{


    double alpha = 0.2;

    double* r = new double[sites_by_id.size()];
    double* r_next = new double[sites_by_id.size()];

    for (size_t i = 0; i < sites_by_id.size(); i++)
    {
        r[i] = 1;
    }
    
    //BEGIN TIMED PORTION
    auto t1 = high_resolution_clock::now();
    int sigma_count = 0;
    // int t_thr;
    // #pragma omp threadprivate(t_thr)

    int row, col_indice, col;
    double sum;
    double sigma = 1;

    #pragma omp parallel shared(r, r_next, row_begin, col_indices, values, row_begin_size, sigma_count, sigma) private(row, col_indice, col, sum)
    {
        while (sigma > 1e-6)
        {
            #pragma omp for
            for(row = 0; row < row_begin_size - 1; row++){
                sum = 0;
                for(col_indice = row_begin[row]; col_indice < row_begin[row+1]; col_indice++){
                    col = col_indices[col_indice];
                    sum += r[col] * values[col_indice];
                }
                r_next[row] = sum * alpha + (1 - alpha);
            }


            #pragma omp single
            {
                sigma = 0;
            }
            
            #pragma omp for reduction(+:sigma)
            for(row = 0; row < row_begin_size - 1; row++){
                sigma = sigma + std::abs(r_next[row] - r[row]);
            }

            #pragma omp single
            {
                std::swap(r, r_next);

                sigma_count++;
                //std::cout << "sigma_"<< sigma_count << " = " << sigma << "\n";
            }
        }
    }

    //END TIMED PORTION
    auto t3 = high_resolution_clock::now();

    int max_indices[] = {-1, -1, -1, -1, -1};
    double max_values[] =  {-1, -1, -1, -1, -1};

    for (size_t i = 0; i < row_begin_size - 1; i++)
    {
        size_t breaking = -1;
        for (size_t j = 0; j < 5; j++)
        {
            if(max_values[j]<=r[i]){
                breaking = j;
            } else {
                break;
            }
        }
        if(breaking != -1){
            for (size_t j = 0; j < breaking; j++)
            {
                max_indices[j] = max_indices[j+1];
                max_values[j] = max_values[j+1];
            }
            max_indices[breaking] = i;
            max_values[breaking] = r[i];
        }
    }

    for (size_t i = 0; i < 5; i++)
    {
        std::cout << sites_by_id[max_indices[i]]->name << ": " << max_values[i] << "\n";
    }
    
    //END TIMED PORTION
    auto t2 = high_resolution_clock::now();

    /* Getting number of milliseconds as an integer. */
    auto ms_int = duration_cast<milliseconds>(t2 - t1);
    auto ms_int2 = duration_cast<milliseconds>(t3 - t1);

    std::cout << ms_int.count() << "ms\n";
    std::cout << ms_int2.count() << "ms\n";
    //std::cout << omp_get_num_threads << "threads\n";
}

void do_row(int row_begin_size, int* row_begin, int* col_indices, double* values)
{
    for (size_t i = 0; i < 8; i++)
    {
        omp_set_num_threads(i);
        timed_portion(row_begin_size, row_begin, col_indices, values);
    }
}

void print_csr(int row_begin_size, int next_row_begin, int* row_begin, int* col_indices, double* values)
{
    std::ofstream rb("csr.txt");

    for (size_t i = 0; i < row_begin_size-1; i++)
    {
        ci << sites_by_id[i]->incoming_edges.size() << "\n";
        //std::cout << row_begin[i] << " ";
    }
    std::cout << std::endl;
    for (size_t i = 0; i < next_row_begin; i++)
    {
        std::cout << values[i] << " ";
    }
    std::cout << std::endl;

    for (size_t i = 0; i < next_row_begin; i++)
    {
        std::cout << col_indices[i] << " ";
    }
    std::cout << std::endl;
}
