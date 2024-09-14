#include <iostream>
#include <vector>
#include <random> 
#include <ctime>
#include <fstream>                                  
#include <sstream>
#include <string>
#include <cmath>
#include <ctime>
using namespace std;

/**
 * 2 layers Neural Network from scratch in C++
 * 
 * @author Dat Le Purdue CS 28
 * @version September 13, 2024
 */

struct matrix {
    vector<vector<double> > data;
    matrix(){}
    matrix(long long rows, int cols) {
        data.resize(rows, vector<double>(cols, 0));
    }

    void print_shape() {
        cout << "Shape: " << data.size() << " x " << data[0].size() << endl;
    }

    matrix operator*(const matrix& other) {
        double epsilon = 1e-14;
        int rows = data.size();
        int cols = other.data[0].size();
        matrix result(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                for (int k = 0; k < other.data.size(); k++) {
                    result.data[i][j] += data[i][k] * other.data[k][j] * 1.0000;
                    if(result.data[i][j] < epsilon) {
                        result.data[i][j] = epsilon; // Avoid log(0)
                    }
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

    matrix operator*(const double a) {
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
                result.data[i][j] = (data[i][j] + other.data[i][j]) * 1.0000;
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
        double epsilon = 1e-9;
        int rows = data.size();
        int cols = data[0].size();
        matrix result(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = (data[i][j] - other.data[i][j]) * 1.0000;
                if(result.data[i][j] < epsilon) {
                    result.data[i][j] = epsilon; // Avoid log(0)
                }
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
                cout << elem << " ";
            }
            cout << endl;
        }
    }

    void print(int n, int m) {
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < m; j++) {
                cout << data[i][j] << " ";
            }
            cout << endl;
        }
    }

    void Randd() {
        for (auto& row : data) {
            for (auto& elem : row) {
                //srand(time(0));
                int a = rand() % 100;
                elem = (double)(a -50) / 100.0000; // Random values between -1 and 1
            }
        }
    }

    matrix ReLU() {
        int rows = data.size();
        int cols = data[0].size();
        matrix result(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = max(0.0, data[i][j]);
            }
        }
        return result;
    }

    matrix softmax() {
        double epsilon = 1e-9;
        int rows = data.size();
        int cols = data[0].size();
        matrix result(rows, cols);
        for (int j = 0; j < cols; j++) {
            double max_val = data[0][j];
            for (int i = 1; i < rows; i++) {
                max_val = max(max_val, data[i][j]);
            }
            double sum = 0;
            for (int i = 0; i < rows; i++) {
                result.data[i][j] = exp(data[i][j] - max_val);
                sum += result.data[i][j];
            }
            for (int i = 0; i < rows; i++) {
                result.data[i][j] /= sum;
                if(result.data[i][j] < epsilon) {
                    result.data[i][j] = epsilon; // Avoid log(0)
                }
            }
        }
        return result;
    }

    double sum() {
        double total = 0;
        for (const auto& row : data) {
            for (const auto& elem : row) {
                total += elem;
            }
        }
        return total;
    }

    matrix argmax_eachcol() {
        int max_index = 0;
        
        matrix freq(10, 1);
        for(int j = 0; j < data[0].size(); j++) {
            double max_value = data[j][0];
            for(int i = 0; i < data.size(); i++) {
                if(data[i][j] > max_value) {
                    max_value = data[i][j];
                    max_index = i;
                }
            }
           freq.data[j][0] = max_index;
        }
        return freq;
    }
};


class NeuralNetwork {
public:
    matrix X_train, Y_train;
    double learning_rate;
    int epochs;
    matrix Z1, Z2, W1, W2, b1, b2, A1, A2, dZ2, dW2, dZ1, dW1;
    double db1, db2;
    NeuralNetwork(matrix X_train, matrix Y_train, double learning_rate, int epochs)
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
        W1.print(10,10);
    }

    void forward() {
        // Forward pass
        //X 784, 42000
        Z1 = (W1 * X_train) + b1;
        A1 = Z1.ReLU(); //10, 42000
        Z2 = (W2 * A1) + b2;
        A2 = Z2.softmax(); //10, 42000
    }

    matrix one_hot_encode(matrix Y) {
        // One hot encoding for Y_train
        matrix one_hot_Y(10, Y.data[0].size());
        for (int i = 0; i < Y.data[0].size(); i++) {
            one_hot_Y.data[Y.data[0][i]][i] = 1;
        }
        return one_hot_Y;
    }

    void backward() {
        // Backward pass
        // Compute gradients and update weights
        // X_train (784, 42000)
        // A1 (10, 42000)
        // A2 (10, 42000)
        // W1 (10, 784)
        // W2 (10, 10)
        // b1 (10, 1)
        // b2 (10, 1)
        double m = Y_train.data[0].size();
        cout<<"m: "<<m<<endl;
        matrix one_hot_Y = one_hot_encode(Y_train);
        dZ2 = A2 - one_hot_Y;
        dW2 = dZ2 * (A1.transpose()) * (1.000/m) ;
        db2 = (1.000/m) * dZ2.sum();
        dZ1 = W2.transpose() * dZ2 * (Z1 > 0);
        dW1 =  dZ1 * (X_train.transpose()) * (1.000/m);
        db1 = (1.000/m) * dZ1.sum();
        dZ2.print_shape();
        dW2.print_shape();
        cout<<db2<<endl;
        dZ1.print_shape();
        dW1.print_shape();
        cout<<db1<<endl;
    };

    void update() {
        // Update weights and biases
        W1 = W1 - (dW1 * learning_rate);
        W2 = W2 - (dW2 * learning_rate);
        b1 = b1 - (db1 * learning_rate);
        b2 = b2 - (db2 * learning_rate);
    }

    matrix predict(matrix X) {
        // Predict the output for a given input
        Z1 = W1 * X + b1;
        A1 = Z1.ReLU();
        Z2 = W2 * A1 + b2;
        A2 = Z2.softmax();
        return A2.argmax_eachcol(); 
    }

    double accuracy(matrix Y) {
        // Calculate accuracy
        int correct = 0;
        matrix predicted=predict(X_train);
        for (int i = 0; i < Y.data[0].size(); i++) {
            if (predicted.data[i][0] == Y.data[0][i]) {
                correct++;
            }
        }
        return (correct * 1.000 / Y.data[0].size()) * 100 * 1.000;
    }

    double mse(matrix Y) {
        // Mean Squared Error
        double error = 0;
        for (int i = 0; i < Y.data[0].size(); i++) {
            error += pow(A2.data[i][0] - Y.data[i][0], 2);
        }
        return error / Y.data[0].size();
    }

    double cross_entropy(matrix Y) {
        // Cross-entropy loss
        double loss = 0;
        for (int i = 0; i < Y.data[0].size(); i++) {
            loss -= Y.data[i][0] * log(A2.data[i][0]);
        }
        return loss / Y.data[0].size();
    }

    void train() {
        for (int i = 0; i < epochs; i++) {
            cout<< "Epoch: " << i + 1 << endl;
            forward();
            cout<<A2.sum()<<endl;
            cout<<"forward pass done"<<endl;
            backward();
            cout<<"backward pass done"<<endl;
            update();
            cout<<"update done"<<endl;
            cout<<accuracy(Y_train)<<"% accuracy after epoch "<<i+1<<endl;
        }
    }
};


matrix one_hot_encode(matrix Y) {
        // One hot encoding for Y_train
        matrix one_hot_Y(10, Y.data[0].size());
        for (int i = 0; i < Y.data[0].size(); i++) {
            one_hot_Y.data[Y.data[0][i]][i] = 1;
        }
        return one_hot_Y;
    }

const int MAX_ROWS = 42000;
const int MAX_COLS = 785;
string data[MAX_ROWS][MAX_COLS];
int main()
{
    ifstream file("data/train.csv");
    
    if (!file.is_open()) {
        cerr << "Error opening file!" << endl;
        return 1;
    }
    // Define a 2D array to store the CSV data
    
    string line;
    int row = 0;
    // Store the CSV data from the CSV file to the 2D array
    while (getline(file, line) && row < MAX_ROWS) {
        stringstream ss(line);
        string cell;
        int col = 0;
        while (getline(ss, cell, ',') && col < MAX_COLS) {
            data[row][col] = cell;
            col++;
        }
        row++;
    }
    //cout<<row;
    // close the file after read opeartion is complete
    file.close();
    cout<<"Succesfully read "<<row<<" rows"<<endl;
    //freopen("data/out.out", "w", stdout);
    matrix X_train(42000, 784);
    for(int i = 0; i < 42000-1; i++) {
        for(int j = 1; j < 785; j++) {
            X_train.data[i][j] = stoi(data[i+1][j]) / 255.0; // Normalize the data
        }
    }
    //cout<<data[42000-1][785-1]<<endl;
    X_train = X_train.transpose();
    matrix Y_train(42000, 1); // read from csv
    for(int i = 0; i < 42000-1; i++) {
        Y_train.data[i][0] = stoi(data[i+1][0]) / 255.0; // Normalize the data
    }
    Y_train = Y_train.transpose();
    // matrix a(2, 3);
    // a.data[0][0] = 1;
    // a.data[0][1] = 2;
    // a.data[0][2] = 3;
    // a.data[1][0] = 6;
    // a.data[1][1] = 7;
    // matrix b(2, 3);
    // b.data[0][0] = 1;
    // b.data[0][1] = 2;
    // b.data[1][0] = 3;
    // b.data[1][1] = 4;

    // a.print();
    // b.print();
    // matrix c = (a>3);
    // c.print();
    //X_train.print(70,27);
    NeuralNetwork net(X_train, Y_train, 0.01, 1);
    net.train();
}

