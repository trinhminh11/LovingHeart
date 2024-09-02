let n_frames = 24;

function preload(){
	mydata = []

	for(let i = 0; i < n_frames; i++){
		mydata.push(loadBytes("frames/" + i + ".bin"));
	}
}

function setup(){
	createCanvas(windowWidth, windowHeight);

	// frame = nj.array(Array.from(mydata[0].bytes)).reshape(1024, 980, 3);

	// frame = Array.from(mydata[0].bytes);

	// console.log(frame.get(0, 0))

	// console.log(frame[0][0][0])


	imgs = []


	for(let t = 0; t < n_frames; t++){
		frame = nj.array(Array.from(mydata[t].bytes)).reshape(1024, 980, 3);

		img = createImage(frame.shape[0], frame.shape[1])
		img.loadPixels();
		for (let i = 0 ; i < frame.shape[0] ; i++){
			for (let j = 0 ; j < frame.shape[1] ; j++){

				let c = [frame.get(i, j, 0), frame.get(i, j, 1), frame.get(i, j, 2)];
				// if (c[0] == 0 && c[1] == 0 && c[2] == 0){
					// continue;
				// }
				img.set(i, j, color(c));
				
			}
		}
		img.updatePixels();
		imgs.push(img);
	}
	count = 0;

	background(0);

	center_x = windowWidth/2;
	center_y = windowHeight/2;

	x = center_x - frame.shape[0]/2;
	y = center_y - frame.shape[1]/2;
}

function draw(){
	frameRate(15);
	
	img = imgs[count];

	image(img, x, y);
	
	count += 1;
	count %= 24;

}

