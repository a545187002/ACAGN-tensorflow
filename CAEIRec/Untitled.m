file_names = dir('*.csv'); %��ȡ���е�.csv�ļ�
for i = 1: numel(file_names)  
    data = csvread(file_names(i).name); %���ζ�ȡcsv��ʽ������
    save([num2str(i),'.mat'],'data');  % save(�ļ���, ����)�����δ洢Ϊmat��ʽ
end 
