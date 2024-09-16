#include <iostream>
#include <vector>
#include <cmath>
using namespace std;


long double pairwise_sum(const std::vector<long double>& data, int start, int end) {
    // Base case: if there is only one element
    if (start == end) {
        return data[start];
    }

    // Find the middle point
    int mid = start + (end - start) / 2;

    // Recursively sum the left and right halves
    long double left_sum = pairwise_sum(data, start, mid);
    long double right_sum = pairwise_sum(data, mid + 1, end);

    // Combine the two sums
    return left_sum + right_sum;
}

// Wrapper function to sum the entire vector
long double sum(const std::vector<long double>& data) {
    return pairwise_sum(data, 0, data.size() - 1);
}

long double kahan_sum(const std::vector<long double>& data) {
    long double total = 0.0;
    long double c = 0.0;  // Compensation for lost low-order bits

    for (const auto& value : data) {
        long double y = value - c;  // Adjust by subtracting the compensation
        long double t = total + y;  // Add the adjusted value
        c = (t - total) - y;  // Compute the new compensation
        total = t;  // Update the total
    }

    return total;
}

int main()
{
    //freopen("data/out.out", "r", stdin);
    freopen("out.txt", "r", stdin);
    long double ans =0.00000000;
    long double x;
    int c=0;
    vector <long double>v;
    while(cin>>x)
    {
        c++;
        v.push_back(x);
        ans+=x;
    }
    cout<<ans<<endl;
    cout<<sum(v)<<" "<<kahan_sum(v)<<endl;
}