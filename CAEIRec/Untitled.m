file_names = dir('*.csv'); %读取所有的.csv文件
for i = 1: numel(file_names)  
    data = csvread(file_names(i).name); %依次读取csv格式的数据
    save([num2str(i),'.mat'],'data');  % save(文件名, 变量)，依次存储为mat格式
end 
