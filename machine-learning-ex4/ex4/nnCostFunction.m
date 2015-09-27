function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

for i = 1: m    
    % add 1 to the beginning of the vector
    % a1: [(input_layer_size + 1) X 1]
    a1 = [1; X(i, :)'];
    
    % compute z2
    % Theta1 size: [hidden_layer_size  X (input_layer_size + 1)], z2 = hidden_layer_size X 1
    z2 = Theta1 * a1;
    
    % compute hidden layer
    % a2 size: hidden_layer_size X 1)
    a2 = sigmoid(z2);
    
    % add 1 to the beginning of a2, size: [(hidden_layer_size + 1) X 1]
    a2 = [1; a2];
    
    % compute the output layer(Theta2: num_laybel X (hidden_layer_dize + 1), h = a3: num_labels X 1, )
    h = sigmoid(Theta2 * a2);
    
    % get the labeled number of ith example
    img_label = y(i:i);
    
    % generate a special vector to compute the cost function
    y_k = zeros(num_labels, 1);
    y_k(img_label, 1) = 1;
    
    % compute the cost of each input example
    cost = - y_k .* log(h) - (1 - y_k) .* log (1 - h);
    
    
    J = J + sum(cost);
    
    % compute gradiant(h: num_labels X 1, y_k: num_labelx X 1)
    delta_3 = h - y_k;
    
    % compute delta_2 for hidden layer. 
    % Theta2': (hidden_layer_size + 1) X num_label, delta3: num_labels X 1, z2: hidden_layer_dize X 1
    delta_2 = Theta2' * delta_3 .* sigmoidGradient([1; z2]);
    
    % removing delta_0
    delta_2 = delta_2(2:end);
    
    % compute Theta2_grad
    % delta3: num_labels X 1, a2' = 1 X (hidden_layer_size + 1), Theta2: num_laybel X (hidden_layer_dize + 1)
    Theta2_grad = Theta2_grad + delta_3 * a2';
    printf ("Theta2_grad size %d X %d\n", size(Theta2_grad));
    
    % compute Theta1_grad
    % delta2: hidden_layer_size X 1, a1': 1 X (input_layer_size + 1), Theta1_grad: [hidden_layer_size  X (input_layer_size + 1)], 
    Theta1_grad = Theta1_grad + delta_2 * a1';
    printf ("Theta1_grad size %d X %d\n", size(Theta1_grad));
    
end
% compte 1/m
discount_factor = 1/m;

% compute the cost    
J = discount_factor * J

% compute gradient of cost
Theta1_grad = discount_factor * Theta1_grad;
Theta2_grad = discount_factor * Theta2_grad;

%compute the regularized cost function 
Theta1_filtered = Theta1(:, 2:end);
Theta2_filtered = Theta2(:, 2:end);
J = J + (lambda / (2 * m)) * (sumsq(Theta1_filtered(:)) + sumsq(Theta2_filtered(:)));


%compute the regularized gradient
Theta1_grad = Theta1_grad + lambda * discount_factor * [zeros(size(Theta1, 1), 1), Theta1(:, 2:end)];
Theta2_grad = Theta2_grad + lambda * discount_factor * [zeros(size(Theta2, 1), 1), Theta2(:, 2:end)];
    
% -------------------------------------------------------------


% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
