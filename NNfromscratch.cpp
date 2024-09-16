#include <iostream>
#include <vector>
#include <random> 
#include <ctime>
#include <fstream>                                  
#include <sstream>
#include <string>
#include <cmath>
#include <ctime>
#include <iomanip>
using namespace std;


/**
 * 2 layers Neural Network from scratch in C++
 * 
 * Due to lack of small floating point precision in C++, the accuracy of the model is not as good as Python. Or I just suck!
 * 
 * @author Dat Le Purdue CS 28
 * @version September 13, 2024
 */


long long Rand(long long l, long long h)
{
    return l + ((long long)rand() * (RAND_MAX + 1) * (RAND_MAX + 1) * (RAND_MAX + 1) +
                (long long)rand() * (RAND_MAX + 1) * (RAND_MAX + 1) +
                (long long)rand() * (RAND_MAX + 1) +
                rand()) % (h - l + 1);
}

struct matrix {
    vector<vector<long double> > data;
    matrix(){}
    matrix(long long rows, int cols) {
        data.resize(rows, vector<long double>(cols, 0.0));
    }

    void print_shape() {
        cout <<"Shape: " << data.size() << " x "  <<  data[0].size() << endl;
    }

    matrix operator*(const matrix& other) {
        if(data[0].size() != other.data.size() ) {
            cout<<"Error: Matrix dimensions do not match"<<endl;
            return *this;
        }
        int rows = data.size();
        int cols = other.data[0].size();
        matrix result(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                for (int k = 0; k < other.data.size(); k++) {
                    result.data[i][j] += data[i][k] * other.data[k][j] * 1.0000;
                }
            }
        }
        return result;
    }

    matrix operator*(const int a) {
        int rows = data.size();
        int cols = data[0].size();
        matrix result(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = data[i][j] * a * 1.0000;
            }
        }
        return result;
    }

    matrix operator*(const long double a) {
        int rows = data.size();
        int cols = data[0].size();
        matrix result(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = data[i][j] * a * 1.0000;
            }
        }
        return result;
    }

    matrix operator+(const matrix& other) {
        int rows = data.size();
        int cols = data[0].size();
        matrix result(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if(i>=other.data.size() || j>=other.data[0].size()) {
                    result.data[i][j] = data[i][j] + other.data[i][0]; //broadcasting
                } else {
                    result.data[i][j] = (data[i][j] + other.data[i][j]) * 1.000;
                }
            }
        }
        return result;
    }

    matrix operator>(const int a) {
        int rows = data.size();
        int cols = data[0].size();
        matrix result(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = data[i][j] > a ? 1 : 0;
            }
        }
        return result;
    }

    matrix operator-(const matrix& other) {
        int rows = data.size();
        int cols = data[0].size();
        matrix result(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if(i>=other.data.size() || j>=other.data[0].size()) {
                    result.data[i][j] = data[i][j] * 1.000;
                } else {
                    result.data[i][j] = (data[i][j] - other.data[i][j]) * 1.000;
                }
            }
        }
        return result;
    }

    matrix operator-(long double a) {
        int rows = data.size();
        int cols = data[0].size();
        matrix result(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = (data[i][j] - a)* 1.0000;
            }
        }
        return result;
    }

    matrix operator-(int a) {
        int rows = data.size();
        int cols = data[0].size();
        matrix result(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = (data[i][j] - a)* 1.0000;
            }
        }
        return result;
    }


    matrix transpose() {
        int rows = data.size();
        int cols = data[0].size();
        matrix result(cols, rows);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[j][i] = data[i][j];
            }
        }
        return result;
    }
    
    void print() {
        for (const auto& row : data) {
            for (const auto& elem : row) {
                cout <<elem << " ";
            }
            cout << endl;
        }
    }

    void print(int n, int m) {
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < m; j++) {
                cout <<fixed<< setprecision(16) << data[i][j] << " ";
            }
            cout << endl;
        }
    }

    void print_end() {
        for(int i = data.size()-1-9; i < data.size(); i++) {
            for(int j = data[0].size()-1-2; j < data[0].size(); j++) {
                cout << data[i][j] << " ";
            }
            cout << endl;
        }
    }

    void Randd() {
        srand((unsigned int)time(NULL));
        int x = 0;
        for (auto& row : data) {
            for (auto& elem : row) {
                int a = Rand(0,3442342) % 100;
                elem = (long double)(a - 50) / 100.0000; // Random values between -1 and 1
            }
        }
    }

    matrix ReLU() {
        int rows = data.size();
        int cols = data[0].size();
        matrix result(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                long double x = 0.0;
                result.data[i][j] = max(x, data[i][j]);
            }
        }
        return result;
    }

    matrix softmax() {
        int rows = data.size();
        int cols = data[0].size();
        matrix result(rows, cols);
        for (int j = 0; j < cols; j++) {
            long double max_val = data[0][j];
            for (int i = 1; i < rows; i++) {
                max_val = max(max_val, data[i][j]);
            }
            long double sum = 0;
            for (int i = 0; i < rows; i++) {
                result.data[i][j] = exp(data[i][j] - max_val);
                sum += result.data[i][j];
            }
            for (int i = 0; i < rows; i++) {
                result.data[i][j] /= sum;
            }
        }
        return result;
    }

    // long double sum() {
    //     long double total = 0;
    //     for (const auto& row : data) {
    //         for (const auto& elem : row) {
    //             total += elem;
    //         }
    //     }
    //     return total;
    // }

    long double pairwise_sum(const std::vector<long double>& data, int start, int end) {
        if (start == end) {
            return data[start];
        }
        int mid = start + (end - start) / 2;
        long double left_sum = pairwise_sum(data, start, mid);
        long double right_sum = pairwise_sum(data, mid + 1, end);
        return left_sum + right_sum;
    }

    long double sum() {
        long double total = 0;
        for (int i = 0; i < data.size(); i++) {
            total += pairwise_sum(data[i], 0, data[i].size() - 1);
        }
        return total;
    }



    matrix argmax_eachcol() {
        int max_index = 0;
        matrix freq(1, data[0].size());
        for(int j = 0; j < data[0].size(); j++) {
            long double max_value = data[0][j];
            for(int i = 0; i < data.size(); i++) {
                if(data[i][j] > max_value) {
                    max_value = data[i][j];
                    max_index = i;
                }
            }
           freq.data[0][j] = max_index;
        }
        return freq;
    }

    matrix mul_elementwise(matrix other) {
        int rows = data.size();
        int cols = data[0].size();
        matrix result(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = data[i][j] * other.data[i][j];
            }
        }
        return result;
    }
};


class NeuralNetwork {
public:
    matrix X_train, Y_train;
    long double learning_rate;
    int epochs;
    matrix Z1, Z2, W1, W2, b1, b2, A1, A2, dZ2, dW2, dZ1, dW1;
    long double db1, db2;
    NeuralNetwork(matrix X_train, matrix Y_train, long double learning_rate, int epochs)
    {
        this->X_train = X_train;
        this->Y_train = Y_train;
        this->learning_rate = learning_rate;
        this->epochs = epochs;
        W1 = matrix(10, 784);
        W2 = matrix(10, 10);
        b1 = matrix(10, 1);
        b2 = matrix(10, 1);
        W1.Randd();
        b1.Randd();
        W2.Randd();
        b2.Randd();
    }

    void forward() {
        Z1 = (W1 * X_train) + b1 ;
        A1 = Z1.ReLU();
        Z2 = (W2 * A1) + b2;
        A2 = Z2.softmax();   
    }

    matrix one_hot_encode(matrix Y) {
        matrix one_hot_Y(10, Y.data[0].size());
        for (int i = 0; i < Y.data[0].size(); i++) {
            one_hot_Y.data[Y.data[0][i]][i] = 1;
        }
        return one_hot_Y;
    }

    void backward() {
        long double m = Y_train.data[0].size();
        matrix one_hot_Y = one_hot_encode(Y_train);
        dZ2 = A2 - one_hot_Y;
        dW2 = dZ2 * (A1.transpose()) * (1.000/m) ;
        db2 = (1.000/m) * dZ2.sum();
        matrix temp = (Z1 > 0);
        dZ1 = W2.transpose() * dZ2;
        dZ1.mul_elementwise(temp);
        dW1 =  dZ1* X_train.transpose()  * (1.000/m);
        db1 = (1.000/m) * dZ1.sum();
    };

    void update() {
        W1 = W1 - (dW1 * learning_rate);  
        W2 = W2 - (dW2 * learning_rate);
        b1 = b1 - (db1 * learning_rate);
        b2 = b2 - (db2 * learning_rate);
    }

    matrix predict(matrix X) {
        Z1 = (W1 * X) + b1 ;
        A1 = Z1.ReLU();
        Z2 = (W2 * A1) + b2;
        A2 = Z2.softmax();
        return A2.argmax_eachcol(); 
    }

    long double accuracy(matrix X) {
        int correct = 0;
        matrix predicted=predict(X);
        for (int i = 0; i < Y_train.data[0].size(); i++) {
            if (predicted.data[0][i] == Y_train.data[0][i]) {
                correct++;
            }
        }
        return (correct * 1.000 /  Y_train.data[0].size()) * 100 * 1.000;
    }

    long double mse(matrix Y) {
        long double error = 0;
        for (int i = 0; i < Y.data[0].size(); i++) {
            error += pow(A2.data[i][0] - Y.data[i][0], 2);
        }
        return error / Y.data[0].size();
    }

    long double cross_entropy(matrix Y) {
        long double loss = 0;
        for (int i = 0; i < Y.data[0].size(); i++) {
            loss -= Y.data[i][0] * log(A2.data[i][0]);
        }
        return loss / Y.data[0].size();
    }

    void train() {
        for (int i = 0; i < epochs; i++) {
            cout<< "Epoch: " << i + 1 << endl;
            forward();
            cout<<"forward pass done"<<endl;
            backward();
            cout<<"backward pass done"<<endl;
            update();
            cout<<"update done"<<endl;
            cout<<accuracy(X_train)<<"% accuracy after epoch "<<i+1<<endl;
        }
    }
};


matrix one_hot_encode(matrix Y) {
        matrix one_hot_Y(10, Y.data[0].size());
        for (int i = 0; i < Y.data[0].size(); i++) {
            one_hot_Y.data[Y.data[0][i]][i] = 1;
        }
        return one_hot_Y;
    }

const int MAX_ROWS = 42001;
const int MAX_COLS = 785;
string dataa[MAX_ROWS][MAX_COLS];
int main()
{
    ifstream file("data/train.csv");
    
    if (!file.is_open()) {
        cerr << "Error opening file!" << endl;
        return 1;
    }
    
    string line;
    int row = 0;
    while (getline(file, line)) {
        stringstream ss(line);
        string cell;
        int col = 0;
        while (getline(ss, cell, ',') && col < MAX_COLS) {
            dataa[row][col] = cell;
            col++;
        }
        row++;
    }
    file.close();
    cout<<"Succesfully read "<<row<<" rows"<<endl;
    matrix X_train(42000, 784);
    for(int i = 0; i < 42000; i++) {
        for(int j = 1; j < 785; j++) {
            X_train.data[i][j-1] = stoi(dataa[i+1][j]) / 255.0; 
        }
    }
    X_train = X_train.transpose();
    matrix Y_train(42000, 1); 
    for(int i = 0; i < 42000; i++) {
        Y_train.data[i][0] = stoi(dataa[i+1][0]); 
    }
    Y_train = Y_train.transpose();
    NeuralNetwork net(X_train, Y_train, 0.3, 1000);
    net.train();
}