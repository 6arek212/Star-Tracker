
# Stars Tracker

In this project we match two images by the stars

<img height="400" src="https://user-images.githubusercontent.com/10331972/233772748-6f8bc387-3a52-415e-883d-02bf070458d3.png"/>


## Try it


    # get star coordinates
    stars1 = finder.get_stars(img1_gray, size)
    stars2 = finder.get_stars(img2_gray, size)

    # write the coordinates to a file
    finder.save_stars_coordinates('./fr1_results.txt', stars1)
    finder.save_stars_coordinates('./fr2_results.txt', stars2)

    # compare imgs
    mapped_stars, source_points, dest_points, line1, points_on_line_1, line2, points_on_line_2 = compare.map_stars(stars1, stars2)

    # display the images
    show_data(source_points, dest_points,
              points_on_line_1, points_on_line_2,  mapped_stars, img1, img2)


## Results

<img  height="400" src="https://user-images.githubusercontent.com/10331972/233772750-708a1bae-8358-4e18-a329-60daae1afe82.png"/>

<img  height="400" src="https://user-images.githubusercontent.com/10331972/233772752-19d1ff79-72e6-4bb8-ae24-e290a7eb5a29.png"/>

<img  height="400" src="https://user-images.githubusercontent.com/10331972/233772755-690c3eff-121c-4e7f-b4ef-b6a72c3dae4f.png"/>


<img  height="400" src="https://user-images.githubusercontent.com/10331972/233788698-4bf377ff-731a-4745-89f9-fcf8e489f047.png"/>


## Our Algorithm

- 1- Run ransac line fitting on stars1 and stars2
- 2- Pick 3 random stars from each set
- 3- make transformation matrix
- 4- return on 2 for n iterations
- return the mapped points , with the best transformation matrix



## Team Members

- Tarik Husin

