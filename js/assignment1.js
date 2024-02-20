import * as THREE from 'three';
import Stats from 'three/addons/libs/stats.module.js';
import { GUI } from 'three/addons/libs/lil-gui.module.min.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';

const container = document.getElementById( 'container1' );
container.style.position = 'relative';
const container2 = document.getElementById('container2');
container2.style.position = 'relative';

let renderer, stats, gui;
let scene, camera, controls, cube, dirlight, ambientLight;
let scene2, camera2, renderer2, controls2, cube2;

let isinitialized = false;

function initScene() {
	scene = new THREE.Scene();
	scene.background = new THREE.Color( 0xffffff);
	camera = new THREE.PerspectiveCamera( 75, window.innerWidth / (window.innerHeight * 0.5), 0.1, 1000 );
	
	renderer = new THREE.WebGLRenderer();
	renderer.setSize( window.innerWidth, window.innerHeight * 0.5 );
	container.appendChild( renderer.domElement );

	controls = new OrbitControls( camera, renderer.domElement );
	controls.minDistance = 2;
	controls.maxDistance = 10;
	controls.addEventListener( 'change', function() { renderer.render( scene, camera ); });

	dirlight = new THREE.DirectionalLight( 0xffffff, 0.5 );
	dirlight.position.set( 0, 0, 1 );
	scene.add( dirlight );

	ambientLight = new THREE.AmbientLight( 0x404040, 2 );
	scene.add( ambientLight );


	// the loading of the object is asynchronous
	let loader = new OBJLoader();
	loader.load( 
		// resource URL
		'../assets/assignment1/cube_subdivided.obj', 
		// called when resource is loaded
		function ( object ) {
			cube = object.children[0];
			cube.material = new THREE.MeshPhongMaterial( { color: 0x999999 });
			cube.position.set( 0, 0, 0 );
			cube.name = "cube";
			scene.add( cube );
		},
		// called when loading is in progresses
		function ( xhr ) {
			console.log( ( xhr.loaded / xhr.total * 100 ) + '% loaded' );
		},
		// called when loading has errors
		function ( error ) {
			console.log( 'An error happened' + error);
		}
	);
	
	camera.position.z = 5;
}
function initScene2() {
    scene2 = new THREE.Scene();
    scene2.background = new THREE.Color(0xffffff);
    camera2 = new THREE.PerspectiveCamera(75, window.innerWidth / (window.innerHeight * 0.5), 0.1, 1000);

    renderer2 = new THREE.WebGLRenderer();
    renderer2.setSize(window.innerWidth, window.innerHeight * 0.5);
    container2.appendChild(renderer2.domElement);

    controls2 = new OrbitControls(camera2, renderer2.domElement);
    controls2.minDistance = 0.1;
    controls2.maxDistance = 0.5;
    controls2.addEventListener('change', function() { renderer2.render(scene2, camera2); });

    let dirlight2 = new THREE.DirectionalLight(0xffffff, 0.5);
    dirlight2.position.set(0, 0, 1);
    scene2.add(dirlight2);

    let ambientLight2 = new THREE.AmbientLight(0x404040, 2);
    scene2.add(ambientLight2);

    let loader2 = new OBJLoader();
    loader2.load(
        '../assets/assignment1/bunny_decimated.obj',
        function(object) {
            cube2 = object.children[0];
            cube2.material = new THREE.MeshPhongMaterial({ color: 0x999999 });
            cube2.position.set(0, 0, 0);
            cube2.name = "cube2";
            scene2.add(cube2);
        },
        function(xhr) {
            console.log((xhr.loaded / xhr.total * 100) + '% loaded');
        },
        function(error) {
            console.log('An error happened' + error);
        }
    );

    camera2.position.z = 5;
}
function initSTATS() {
	stats = new Stats();
	stats.showPanel( 0 );
	stats.dom.style.position = 'absolute';
	stats.dom.style.top = 0;
	stats.dom.style.left = 0;
	container.appendChild( stats.dom );
}

function initGUI() {
	if (!isinitialized) {
		gui = new GUI();
		cube = scene.getObjectByName( "cube" );
		gui.add( cube.position, 'x', -1, 1 );
		gui.add( cube.position, 'y', -1, 1 );
		gui.add( cube.position, 'z', -1, 1 );
		gui.domElement.style.position = 'absolute';
		gui.domElement.style.top = '0px';
		gui.domElement.style.right = '0px';
		container.appendChild( gui.domElement );
		isinitialized = true;
	}
}

function animate() {
    requestAnimationFrame(animate);

    cube = scene.getObjectByName("cube");
    if (cube) {
        cube.rotation.x += 0.01;
        cube.rotation.y += 0.01;
        initGUI(); // initialize the GUI after the object is loaded
    }

    renderer.render(scene, camera);
    stats.update();

    // Render the second scene
	cube2 = scene.getObjectByName("cube2");
    if (cube2) {
        cube2.rotation.x += 0.01;
        cube2.rotation.y += 0.01;
    }
    renderer2.render(scene2, camera2);
}

function onWindowResize() {
	camera.aspect = window.innerWidth / (window.innerHeight * 0.5);
	camera.updateProjectionMatrix();
	renderer.setSize( window.innerWidth, window.innerHeight * 0.5 );
	camera2.aspect = window.innerWidth / (window.innerHeight * 0.5);
    camera2.updateProjectionMatrix();
    renderer2.setSize(window.innerWidth, window.innerHeight * 0.5);
};

window.addEventListener( 'resize', onWindowResize, false );

initScene();
initScene2();

initSTATS();
animate();
