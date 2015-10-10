function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returs the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

%Create a verctor to track hom many example are assigned to the centroid
cent_num_points = zeros(K, 1);

% Regenerate distance for each centroid by scanning all example and add them to the centroid they are assigned.
for i = 1:m
  cent_assigned = idx(i, 1);
  if cent_assigned > K || cent_assigned < 1
    continue
  endif
  centroids(cent_assigned, :) = centroids(cent_assigned, :) + X(i, :);
  cent_num_points(cent_assigned, 1) = cent_num_points(cent_assigned, 1) + 1;
endfor

% Divid each centroid by the number of points assigned to them
for k = 1:K
  num_points = cent_num_points(k, 1);
  if num_points > 0
    centroids(k, :) = (1 / num_points) * centroids(k, :);
  endif
endfor

% =============================================================


end

