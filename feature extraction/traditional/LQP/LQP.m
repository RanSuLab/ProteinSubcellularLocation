function [ ltp_upper1, ltp_lower1,ltp_upper2, ltp_lower2 ] = LQP(im, t1,t2)

    %// Get the dimensions
    rows=size(im,1);
    cols=size(im,2);

    %// Reordering vector - Essentially for getting binary strings
    reorder_vector = [8 7 4 1 2 3 6 9];

    %// For the upper and lower LTP patterns
    ltp_upper1 = zeros(size(im));
    ltp_lower1 = zeros(size(im));
    ltp_upper2 = zeros(size(im));
    ltp_lower2 = zeros(size(im));
    %// For each pixel in our image, ignoring the borders...
    for row = 2 : rows - 1
        for col = 2 : cols - 1
            cen = im(row,col); %// Get centre

            %// Get neighbourhood - cast to double for better precision
            pixels = double(im(row-1:row+1,col-1:col+1));

            %// Get ranges and determine LTP
            out_LTP = zeros(3, 3);
            low1 = cen - t1;
            high1 = cen + t1;
            low2 = cen - t2;
            high2 = cen + t2;
            out_LTP(pixels > low2 & pixels < low1) = -1;
            out_LTP(pixels > high1 & pixels < high2) = 1;
            out_LTP(pixels < low2) = -2;
            out_LTP(pixels >= high2) = 2;
	        out_LTP(pixels >= low1 & pixels < high1) = 0;

            %// Get upper and lower patterns
            upper1 = out_LTP;
            upper1(upper1 == 1) = 1;
            upper1(upper1 == -1) = 0;
            upper1(upper1 == -2) = 0;
            upper1(upper1 == 2) = 0;
            upper1 = upper1(reorder_vector);

            lower1 = out_LTP;
            lower1(lower1 == 1) = 0;
            lower1(lower1 == -1) = 1;
            lower1(lower1 == -2) = 0;
            lower1(lower1 == 2) = 0;
            lower1 = lower1(reorder_vector);
            
            upper2 = out_LTP;
            upper2(upper2 == -1) = 0;
            upper2(upper2 == 1) = 0;
            upper2(upper2 == -2) = 0;
            upper2(upper2 == 2) = 1;
            upper2 = upper2(reorder_vector);

            lower2 = out_LTP;
            lower2(lower2 == 1) = 0;
            lower2(lower2 == -1) = 0;
            lower2(lower2 == -2) = 1;
            lower2(lower2 == 2) = 0;          
            lower2 = lower2(reorder_vector);
            %// Convert to a binary character string, then use bin2dec
            %// to get the decimal representation
            upper_bitstring = char(48 + upper1);
            ltp_upper1(row,col) = bin2dec(upper_bitstring);

            lower_bitstring = char(48 + lower1);
            ltp_lower1(row,col) = bin2dec(lower_bitstring);%将二进制转换为十进制
            
            upper_bitstring = char(48 + upper2);
            ltp_upper2(row,col) = bin2dec(upper_bitstring);

            lower_bitstring = char(48 + lower2);
            ltp_lower2(row,col) = bin2dec(lower_bitstring);%将二进制转换为十进制
       end
   end