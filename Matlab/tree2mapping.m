function real_mapping = tree2mapping(tree_vector, densenum)
% This function transform a tree to a mapping matrix
vertex_num = size(tree_vector, 1);
total_mapping = zeros(vertex_num, 100);
valid = zeros(vertex_num, 1);
pooling_vertex_num = zeros(vertex_num, 1);

simpnum = 0;
for i = 1:vertex_num
    if ~tree_vector(i,1)
        valid(i,1) = 1;
        simpnum = simpnum + 1;
    end
end

for i = 1:densenum
    j = i;
    while tree_vector(j,1)~=0
        col_num = pooling_vertex_num(j, 1) + 1;
        total_mapping(j,col_num) = i;
        pooling_vertex_num(j,1) = pooling_vertex_num(j,1) + 1;
        j = tree_vector(j);
    end
    col_num = pooling_vertex_num(j, 1) + 1;
    total_mapping(j,col_num) = i;
    pooling_vertex_num(j,1) = pooling_vertex_num(j,1) + 1;
end

row = 1;
max_pooling_num = 0;
real_mapping = zeros(simpnum, 100);
for i = 1:size(valid,1)
    if valid(i,1)
        for j = 1:pooling_vertex_num(i,1)
            real_mapping(row,j) = total_mapping(i,j);
        end
        if max_pooling_num < pooling_vertex_num(i,1)
            max_pooling_num = pooling_vertex_num(i,1);
        end
        row = row + 1;
    end
end

real_mapping(:,max_pooling_num+1:end) = [];

end